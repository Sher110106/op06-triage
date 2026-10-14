"""Build the prediction files the taxonomy needs that no earlier run produced, and grade each one.

Every run reads the projected inputs of dev + dev_noisy + hostile (2,309 rows) and the shipped CPU model
(models/classical_v0.joblib). Shipped code is not edited: each change is a constructor argument, a rebuild
from bundle parts, or a monkeypatch that lives only in this process.

  taxonomy-classical-nodedupe  the classical model with sentence de-duplication off (Predictor(dedupe=False))
  taxonomy-rule-stage          the probe's "rule" stage rebuilt from the bundle: the bundle's TF-IDF intent and
                               flow models, then (predicted intent, last observed action) -> most common next
                               action on train fold A, backing off to (flow of predicted intent, last action),
                               (last action) and the global majority, then the soft policy prior with the alias.
                               No de-duplication, as in the probe (it predates R6).
  taxonomy-classical-nocap     exploratory: v0 with the 2,000-character turn cap off (dedupe stays on)
  taxonomy-classical-nodedupe-nocap
                               exploratory: v0 with both de-duplication and the turn cap off
  taxonomy-classical-nonfkc    exploratory: v0 with NFKC normalisation off
  taxonomy-classical-nozw      exploratory: v0 with zero-width and bidi format-character removal off

  python experiments/taxonomy/build_runs.py [--only RUN ...] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from triage import text as T  # noqa: E402
from triage.classical import DEFAULT_BUNDLE, Predictor, apply_prior, folds  # noqa: E402
from triage.data import DATA, load_split, project  # noqa: E402

RAW = ROOT / "results" / "raw"
PY = sys.executable


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs():
    rows = load_split("dev") + load_split("noisy") + load_split("hostile")
    return [project(r) for r in rows]


@contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


class _NoNFKC:
    """Stands in for the unicodedata module inside triage.text only: normalize() returns the text unchanged."""

    @staticmethod
    def normalize(form, s):
        return s


# ----------------------------------------------------------------------------- systems

def classical(dedupe=True):
    pm = Predictor(DEFAULT_BUNDLE, dedupe=dedupe)
    return lambda reqs: pm.predict_many(reqs)


def classical_patched(name, value, dedupe=True):
    pm = Predictor(DEFAULT_BUNDLE, dedupe=dedupe)

    def run(reqs):
        with patched(T, name, value):
            return pm.predict_many(reqs)
    return run


def rule_stage(reqs):
    """The probe's rule stage (e4_action.py, rule_predintent_lastobs with the soft alias prior), on bundle parts."""
    pm = Predictor(DEFAULT_BUNDLE, dedupe=False)
    b = pm.b
    acts = list(b["acts"])
    aidx = {a: j for j, a in enumerate(acts)}
    A, B = folds()
    intent_flow = {r["intent"]: r["flow"] for r in A + B}
    tab1, tab2, tab3 = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for r in A:
        lo = T.last_obs(project(r))
        tab1[(r["intent"], lo)][r["action"]] += 1
        tab2[(r["flow"], lo)][r["action"]] += 1
        tab3[lo][r["action"]] += 1
    glob = Counter(r["action"] for r in A)

    rows = [pm._sanitise(q) for q in reqs]
    out = [None] * len(rows)
    live = []
    for i, r in enumerate(rows):
        if not any(re.search(r"[a-z0-9]", T.clean_text(t["text"])) for t in r["context"]):
            out[i] = Predictor.abstain(r["id"], "no readable text in context")
        else:
            live.append(i)
    lrows = [rows[i] for i in live]
    _, pi, pf = pm.raw_probs(lrows)
    intents = [str(pm.icls[k]) for k in pi.argmax(1)]
    P = np.zeros((len(lrows), len(acts)))
    for n, (r, it) in enumerate(zip(lrows, intents)):
        lo = T.last_obs(r)
        cnt = tab1.get((it, lo)) or tab2.get((intent_flow[it], lo)) or tab3.get(lo) or glob
        tot = sum(cnt.values())
        for a, v in cnt.items():
            P[n, aidx[a]] = v / tot
    P = apply_prior(P, pf, b["perm"])
    for n, i in enumerate(live):
        c = float(min(1.0, max(0.0, round(float(P[n].max()), 6))))
        out[i] = {"id": rows[i]["id"], "intent": intents[n], "action": acts[int(P[n].argmax())],
                  "confidence": c, "needs_human": bool(c < 0.5)}
    return out


RUNS = {
    "taxonomy-classical-nodedupe": dict(
        fn=lambda: classical(dedupe=False), base="dev-classical-v0",
        changed="triage.classical.Predictor(models/classical_v0.joblib, dedupe=False): sentence de-duplication off; "
                "everything else as dev-classical-v0",
        status="required column (b) of the taxonomy"),
    "taxonomy-rule-stage": dict(
        fn=lambda: rule_stage, base="dev-baseline",
        changed="probe-era rule stage rebuilt from the shipped bundle: bundle TF-IDF LinearSVC intent and flow "
                "models (no de-duplication), next action = most common action on train fold A for (predicted "
                "intent, last observed action), backing off to (flow of predicted intent, last action), (last "
                "action), global majority; soft policy prior with the membership alias (bundle 'perm'). "
                "confidence = the rule's p(action) after the prior; needs_human = confidence < 0.5 (not tuned, "
                "not meaningful)",
        status="optional column: the 13 Oct probe's rule stage (EXPERIMENT_LOG R4), rebuilt; not a shipped system"),
    "taxonomy-classical-nocap": dict(
        fn=lambda: classical_patched("TURN_CAP", 10 ** 9), base="dev-classical-v0",
        changed="triage.text.TURN_CAP monkeypatched from 2000 to 1e9 in this process: no per-turn length cap; "
                "dedupe on; model unchanged",
        status="exploratory ablation, not pre-registered"),
    "taxonomy-classical-nodedupe-nocap": dict(
        fn=lambda: classical_patched("TURN_CAP", 10 ** 9, dedupe=False), base="taxonomy-classical-nodedupe",
        changed="Predictor(dedupe=False) and triage.text.TURN_CAP monkeypatched from 2000 to 1e9: neither "
                "de-duplication nor the per-turn cap; model unchanged",
        status="exploratory ablation, not pre-registered"),
    "taxonomy-classical-nonfkc": dict(
        fn=lambda: classical_patched("unicodedata", _NoNFKC), base="dev-classical-v0",
        changed="triage.text.unicodedata monkeypatched so NFKC normalisation is skipped in clean_text; model "
                "unchanged",
        status="exploratory ablation, not pre-registered"),
    "taxonomy-classical-nozw": dict(
        fn=lambda: classical_patched("_ZW", re.compile(r"(?!)")), base="dev-classical-v0",
        changed="triage.text._ZW monkeypatched to a pattern that matches nothing: zero-width and bidi format "
                "characters are no longer removed in clean_text; model unchanged",
        status="exploratory ablation, not pre-registered"),
}


def build(run_id: str, force: bool):
    spec = RUNS[run_id]
    out = RAW / run_id
    if out.exists() and not force:
        print(f"{run_id}: exists, skipped (pass --force to rebuild)")
        return
    out.mkdir(parents=True, exist_ok=True)
    reqs = inputs()
    t0 = time.perf_counter()
    preds = spec["fn"]()(reqs)
    wall = time.perf_counter() - t0
    assert len(preds) == len(reqs) and all(p["id"] == q["id"] for p, q in zip(preds, reqs))
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    meta = {"run_id": run_id, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "system": "classical (experiment)",
            "what_changed": spec["changed"], "compare_with": f"results/raw/{spec['base']}", "status": spec["status"],
            "bundle": str(DEFAULT_BUNDLE.relative_to(ROOT)), "bundle_sha256": sha256(DEFAULT_BUNDLE),
            "script": str(Path(__file__).resolve().relative_to(ROOT)), "script_sha256": sha256(__file__),
            "inputs": {k: {"file": f"data/op06/{f}", "sha256": sha256(DATA / f)}
                       for k, f in (("dev", "dev.jsonl"), ("noisy", "dev_noisy.jsonl"), ("hostile", "hostile.jsonl"))},
            "rows": len(reqs), "batch_wall_s": round(wall, 2), "python": platform.python_version(),
            "machine": f"{platform.system()} {platform.machine()}", "llm_calls": 0,
            "note": "batch prediction over the projected inputs (id, turn_index, context); no API call"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    subprocess.run([PY, "-m", "harness.eval", "grade", "--pred", str(out / "predictions.jsonl")], cwd=ROOT, check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=sorted(RUNS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    for run_id in a.only or RUNS:
        build(run_id, a.force)


if __name__ == "__main__":
    main()
