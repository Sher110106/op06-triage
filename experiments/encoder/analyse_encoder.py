"""Score the encoder comparison exactly as PREREGISTRATION.md fixes it. Inputs: the remote run's cache files
(copied into results/raw/encoder-ett32A/; they lived in experiments/encoder/ett32A/ until 14 Oct), the classical
bundle, the fold-B CPU files from ../llm_confirm/ and the fold-B Luna answers (results/raw/llm-confirm-luna_B/).

  python experiments/encoder/analyse_encoder.py [--cache results/raw/encoder-ett32A]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "llm_confirm"))
from harness.eval import paired, read_preds  # noqa: E402
from triage.classical import DEFAULT_BUNDLE, Predictor, folds  # noqa: E402
from triage.data import load_split, project  # noqa: E402

GRID = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]


def enc_probs(cache, split, head, labels, target):
    P = np.load(cache / f"ft_ett32A__{head}__{split}.npy")
    ids = json.loads((cache / f"ft_ett32A__ids__{split}.json").read_text())
    col = {c: j for j, c in enumerate(labels[head])}
    out = np.zeros((len(ids), len(target)))
    for k, c in enumerate(target):
        if c in col:
            out[:, k] = P[:, col[c]]
    return ids, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/raw/encoder-ett32A", help="folder, relative to the repo root")
    a = ap.parse_args()
    cache = ROOT / a.cache
    labels = json.loads((cache / "labels.json").read_text())
    pm = Predictor(DEFAULT_BUNDLE)
    icls, acts = list(pm.icls), list(pm.acts)

    def classical(rows, dedupe):
        pm.dedupe = dedupe
        reqs = [pm._sanitise(project(r)) for r in rows]
        pa, pi, _ = pm._probs(reqs)
        return pa, pi

    dev = load_split("dev") + load_split("noisy")
    _, B = folds()
    pa_d, pi_d = classical(dev, True)
    pa_b, pi_b = classical(B, False)           # as in llm_confirm/prep.py
    ids_d = []
    ea_d, ei_d = [], []
    for split in ("dev", "noisy"):
        i1, a1 = enc_probs(cache, split, "action", labels, acts)
        _, b1 = enc_probs(cache, split, "intent", labels, icls)
        ids_d += i1; ea_d.append(a1); ei_d.append(b1)
    ea_d, ei_d = np.vstack(ea_d), np.vstack(ei_d)
    assert ids_d == [r["id"] for r in dev]
    ids_b, ea_b = enc_probs(cache, "foldB", "action", labels, acts)
    _, ei_b = enc_probs(cache, "foldB", "intent", labels, icls)
    assert ids_b == [r["id"] for r in B]

    gold_a = np.array([r["action"] for r in dev]); gold_i = np.array([r["intent"] for r in dev])
    acc = lambda P, cls, gold: float(np.mean(np.array(cls)[P.argmax(1)] == gold))
    w = max(GRID, key=lambda w: acc(w * pa_d + (1 - w) * ea_d, acts, gold_a))
    v = max(GRID, key=lambda v: acc(v * pi_d + (1 - v) * ei_d, icls, gold_i))
    dev_table = {f"w={g}": acc(g * pa_d + (1 - g) * ea_d, acts, gold_a) for g in GRID}

    c0 = read_preds(ROOT / "experiments/llm_confirm/cpu_B_predictions.jsonl")
    det = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "experiments/llm_confirm/cpu_B_details.jsonl")}
    import analyse_B as AB
    luna, _ = AB.answers(ROOT / "results/raw/llm-confirm-luna_B/raw_calls.jsonl")
    s0 = AB.system(c0, det, act_src=luna)

    def system(PA, PI):
        out = {}
        for k, rid in enumerate(ids_b):
            q = dict(c0[rid])
            q["intent"] = icls[int(PI[k].argmax())]
            q["action"] = acts[int(PA[k].argmax())]
            out[rid] = q                        # confidence kept from C0: this test compares accuracy only
        return out

    e1 = system(ea_b, ei_b)
    e2 = system(w * pa_b + (1 - w) * ea_b, v * pi_b + (1 - v) * ei_b)
    rows = [dict(r, _group="clean") for r in B]
    keys = ("intent_acc", "macro_f1", "action_acc")
    res = {"weights_from_dev": {"action_w_classical": w, "intent_v_classical": v}, "dev_action_by_w": dev_table,
           "dev_encoder_alone": {"intent": acc(ei_d, icls, gold_i), "action": acc(ea_d, acts, gold_a)},
           "dev_classical": {"intent": acc(pi_d, icls, gold_i), "action": acc(pa_d, acts, gold_a)}}
    for name, sysp in (("E1_vs_C0", e1), ("E2_vs_C0", e2), ("S0_vs_C0", s0)):
        res[name] = paired(c0, sysp, rows=rows, keys=keys)["metrics"]
    for name, sysp in (("E1_vs_S0", e1), ("E2_vs_S0", e2)):
        res[name] = paired(s0, sysp, rows=rows, keys=keys)["metrics"]
    m = res["E2_vs_C0"]["action_acc"]; n = res["E2_vs_S0"]["action_acc"]
    res["decision"] = {
        "real_cpu_gain": m["diff"] >= 0.02 and m["ci95"][0] > 0,
        "replace_llm_step_in_v1": n["diff"] >= 0.02 and n["ci95"][0] > 0,
        "v2_path_non_inferior": n["ci95"][0] > -0.005,
    }
    (HERE / "results_encoder.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["decision"]), json.dumps(res["weights_from_dev"]))
    for k in ("E1_vs_C0", "E2_vs_C0", "S0_vs_C0", "E1_vs_S0", "E2_vs_S0"):
        print(k, {x: (res[k][x]["new"], res[k][x]["diff"], res[k][x]["ci95"]) for x in keys})


if __name__ == "__main__":
    main()
