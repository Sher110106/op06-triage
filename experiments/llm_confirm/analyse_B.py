"""Score the fold-B confirmation test exactly as PREREGISTRATION.md fixes it. No API calls.

  python analyse_B.py --luna runs/luna_B.jsonl --haiku runs/haiku_B.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from harness.eval import paired, read_preds  # noqa: E402
from triage.classical import folds  # noqa: E402

T20 = 0.642841          # 20% quantile of CPU confidence on dev (fixed before this test)
P_GATE, TOP_K = 0.95, 3  # intent rule chosen on dev by 5-fold CV (rule_selection_dev.json)
THR = 0.8498            # CPU routing threshold
MIN_GAIN = 0.020


def answers(path):
    out, st = {}, {"rows": 0, "usable": 0, "failed_or_invalid": 0, "abstained": 0}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        st["rows"] += 1
        p = r.get("pred")
        if r.get("status") != "ok" or r.get("parse_status") not in ("ok", "lenient_ok") or not p:
            st["failed_or_invalid"] += 1
        elif p["intent"] == "unknown" or p["action"] == "none":
            st["abstained"] += 1
        else:
            out[r["id"]] = (p["intent"], p["action"])
            st["usable"] += 1
    return out, st


def calls(path, ids=None):
    recs = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]
    if ids is not None:
        recs = [r for r in recs if r["id"] in ids]
    lat = sorted(r["latency_s"] for r in recs if r.get("status") == "ok")
    pct = lambda q: round(lat[min(len(lat) - 1, int(math.ceil(q * len(lat))) - 1)], 3) if lat else None
    return {"calls": len(recs), "p50_s": pct(0.5), "p95_s": pct(0.95),
            "cost_list_with_cache": round(sum(r.get("cost_list_actual") or 0 for r in recs), 4),
            "cost_list_uncached": round(sum(r.get("cost_list_uncached") or 0 for r in recs), 4)}


def system(cpu, det, act_src=None, int_src=None):
    out = {}
    for rid, p in cpu.items():
        q = dict(p)
        if int_src is not None and det[rid]["intent_top5"][0][1] < P_GATE and rid in int_src:
            h = int_src[rid][0]
            if h in [x for x, _ in det[rid]["intent_top5"][:TOP_K]]:
                q["intent"] = h
        if act_src is not None and p["confidence"] < T20 and rid in act_src:
            a = act_src[rid][1]
            q["action"] = a
            q["confidence"] = round((p["confidence"] + (1.0 if a == p["action"] else 0.0)) / 2, 6)
            q["needs_human"] = bool(q["confidence"] < THR)
        out[rid] = q
    return out


def passes(m, key):
    return m[key]["diff"] >= MIN_GAIN and m[key]["ci95"][0] > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--luna", required=True)
    ap.add_argument("--haiku", required=True)
    a = ap.parse_args()
    _, B = folds()
    rows = [dict(r, _group="clean") for r in B]
    cpu = read_preds(HERE / "cpu_B_predictions.jsonl")
    det = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "cpu_B_details.jsonl")}
    luna, lst = answers(a.luna)
    haiku, hst = answers(a.haiku)
    keys = ("intent_acc", "macro_f1", "action_acc", "top70_error_ratio")
    systems = {"L_act": system(cpu, det, act_src=luna), "L_int": system(cpu, det, int_src=luna),
               "H_int": system(cpu, det, int_src=haiku), "L_both": system(cpu, det, act_src=luna, int_src=luna)}
    res = {"rows": len(rows), "T20": T20, "share_below_T20": round(float(np.mean([p["confidence"] < T20 for p in cpu.values()])), 4),
           "share_intent_gate": round(float(np.mean([d["intent_top5"][0][1] < P_GATE for d in det.values()])), 4),
           "luna_answers": lst, "haiku_answers": hst, "systems": {}}
    for name, preds in systems.items():
        res["systems"][name] = paired(cpu, preds, rows=rows, keys=keys)
    s = {k: v["metrics"] for k, v in res["systems"].items()}
    act = passes(s["L_act"], "action_acc")
    l_int, h_int = passes(s["L_int"], "intent_acc"), passes(s["H_int"], "intent_acc")
    intent_step = "luna" if l_int else ("haiku" if h_int else None)
    res["decision"] = {"action_step_luna": act, "intent_step_luna": l_int, "intent_step_haiku": h_int,
                       "v1": {"action_step": act, "intent_step": intent_step}}
    # what the adopted Luna-only system calls, and costs, per 1,000 messages
    gate = {rid for rid, p in cpu.items() if p["confidence"] < T20 or det[rid]["intent_top5"][0][1] < P_GATE}
    res["luna_union_rows"] = {"share": round(len(gate) / len(cpu), 4), **calls(a.luna, gate)}
    res["luna_all_rows"] = calls(a.luna)
    res["haiku_gated_rows"] = calls(a.haiku)
    (HERE / "results_B.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["decision"]))
    for name, m in s.items():
        print(name, {k: (m[k]["new"], m[k]["diff"], m[k]["ci95"]) for k in keys})


if __name__ == "__main__":
    main()
