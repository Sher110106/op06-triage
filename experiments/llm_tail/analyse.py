"""Score the full-dev LLM test exactly as PREREGISTRATION.md fixes it.

  python experiments/llm_tail/analyse.py      # the defaults below are the files the 14 Oct analysis used:
      --cpu results/raw/dev-classical-v0/predictions.jsonl
      --luna results/raw/llm-tail-luna_V3_none/raw_calls.jsonl --haiku results/raw/llm-tail-haiku_V3_low/raw_calls.jsonl
(The raw call logs lived in experiments/llm_tail/runs/ until 14 Oct; experiments/relocate/moves.json.)

Writes system prediction files and reports under ../../results/raw/llm-tail-<system>/ and a summary to
results.json next to this file. Nothing here calls an API.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from harness.eval import RAW, paired, read_preds  # noqa: E402

THR = 0.8498            # CPU routing threshold, fixed on train hold-out B (models/classical_v0.joblib meta)
MIN_GAIN = 0.020


def llm_answers(path):
    """id -> (intent, action) for usable answers only (the fallback rule in PREREGISTRATION.md)."""
    out, stats = {}, {"rows": 0, "usable": 0, "failed_or_invalid": 0, "abstained": 0}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        stats["rows"] += 1
        p = r.get("pred")
        if r.get("status") != "ok" or r.get("parse_status") not in ("ok", "lenient_ok") or not p:
            stats["failed_or_invalid"] += 1
            continue
        if p["intent"] == "unknown" or p["action"] == "none":
            stats["abstained"] += 1
            continue
        out[r["id"]] = (p["intent"], p["action"])
        stats["usable"] += 1
    return out, stats


def run_stats(path, n_messages=1000):
    recs = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]
    ok = [r for r in recs if r.get("status") == "ok"]
    lat = sorted(r["latency_s"] for r in ok)
    pct = lambda q: round(lat[min(len(lat) - 1, int(math.ceil(q * len(lat))) - 1)], 3) if lat else None
    per = lambda k: sum((r.get(k) or 0) for r in recs) / max(1, len(recs)) * n_messages
    return {"calls": len(recs), "ok": len(ok), "p50_s": pct(0.5), "p95_s": pct(0.95),
            "cost_per_1000_list_uncached": round(per("cost_list_uncached"), 4),
            "cost_per_1000_list_with_cache": round(per("cost_list_actual"), 4),
            "models_returned": sorted({str(r.get("model_returned")) for r in recs}),
            "providers_returned": sorted({str(r.get("provider_returned")) for r in recs})}


def build(cpu, intents=None, tail=None, escalate=None):
    """intents: id -> intent used on every row (B). tail: id -> (intent, action) used on escalated rows (A)."""
    out = {}
    for rid, p in cpu.items():
        q = dict(p)
        if rid.startswith("hostile"):
            out[rid] = q
            continue
        if intents and rid in intents:
            q["intent"] = intents[rid][0]
        if tail is not None and escalate(rid, p) and rid in tail:
            it, ac = tail[rid]
            if not intents:
                q["intent"] = it
            q["action"] = ac
            agree = 1.0 if ac == p["action"] else 0.0
            q["confidence"] = round((p["confidence"] + agree) / 2, 6)
            q["needs_human"] = bool(q["confidence"] < THR)
        out[rid] = q
    return out


def write(name, preds):
    d = RAW / f"llm-tail-{name}"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "predictions.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for p in preds.values():
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    return path


def grade(path):
    import subprocess
    subprocess.run([sys.executable, "-m", "harness.eval", "grade", "--pred", str(path)], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    return json.loads(path.with_name("report.json").read_text())


def verdict(cmp, keys):
    best = None
    for k in keys:
        m = cmp["metrics"][k]
        ok = m["diff"] >= MIN_GAIN and m["ci95"][0] > 0
        best = best or ok
    return bool(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", default=str(ROOT / "results/raw/dev-classical-v0/predictions.jsonl"))
    ap.add_argument("--luna", default=str(ROOT / "results/raw/llm-tail-luna_V3_none/raw_calls.jsonl"))
    ap.add_argument("--haiku", default=str(ROOT / "results/raw/llm-tail-haiku_V3_low/raw_calls.jsonl"))
    a = ap.parse_args()
    cpu = read_preds(a.cpu)
    seen = set(json.loads((HERE / "seen_in_probe_ids.json").read_text()))
    luna, luna_st = llm_answers(a.luna)
    haiku, haiku_st = llm_answers(a.haiku)
    routed = lambda rid, p: p["confidence"] < THR

    systems = {"A_luna_tail": build(cpu, tail=luna, escalate=routed),
               "B_haiku_intent": build(cpu, intents=haiku),
               "C_both": build(cpu, intents=haiku, tail=luna, escalate=routed)}
    res = {"threshold": THR, "min_gain": MIN_GAIN, "luna_answers": luna_st, "haiku_answers": haiku_st,
           "luna_calls": run_stats(a.luna), "haiku_calls": run_stats(a.haiku),
           "escalated_share": round(float(np.mean([routed(r, p) for r, p in cpu.items() if not r.startswith("hostile")])), 4),
           "systems": {}}
    for name, preds in systems.items():
        path = write(name, preds)
        rep = grade(path)
        fresh = paired(cpu, preds, exclude=seen)
        full = paired(cpu, preds)
        res["systems"][name] = {"pred_file": str(path.relative_to(ROOT)), "bars_pass_all_rows": all(v for v in rep["bars"].values() if v is not None),
                                "official_all_rows": rep["official"], "paired_fresh": fresh, "paired_all": full}
    s = res["systems"]
    passA = verdict(s["A_luna_tail"]["paired_fresh"], ["action_acc"]) and s["A_luna_tail"]["bars_pass_all_rows"]
    passB = verdict(s["B_haiku_intent"]["paired_fresh"], ["intent_acc", "macro_f1"]) and s["B_haiku_intent"]["bars_pass_all_rows"]
    passC = (verdict(s["C_both"]["paired_fresh"], ["action_acc"]) and verdict(s["C_both"]["paired_fresh"], ["intent_acc", "macro_f1"])
             and s["C_both"]["bars_pass_all_rows"])
    res["decision"] = {"adopt_A": passA, "adopt_B": passB, "adopt_C": passC,
                       "v1_llm": "C" if passC else ("A" if passA else ("B" if passB else "none"))}

    # exploratory (not part of the decision): other escalation shares by CPU confidence rank, and Luna for intent
    lab = {r: p for r, p in cpu.items() if not r.startswith("hostile")}
    order = sorted(lab, key=lambda r: lab[r]["confidence"])
    explo = {}
    for share in (0.1, 0.2, 0.3, 0.4, 0.5, 1.0):
        chosen = set(order[: int(round(share * len(order)))])
        preds = build(cpu, tail=luna, escalate=lambda rid, p, c=chosen: rid in c)
        m = paired(cpu, preds, exclude=seen)["metrics"]
        explo[f"luna_tail_{int(share * 100)}pct"] = {k: (m[k]["new"], m[k]["diff"], m[k]["ci95"])
                                                     for k in ("intent_acc", "macro_f1", "action_acc", "noisy_gap", "top70_error_ratio")}
    m = paired(cpu, build(cpu, intents=luna), exclude=seen)["metrics"]
    explo["luna_intent_all"] = {k: (m[k]["new"], m[k]["diff"], m[k]["ci95"]) for k in ("intent_acc", "macro_f1")}
    res["exploratory_fresh_rows"] = explo
    (HERE / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["decision"]))
    for name, v in s.items():
        print(name, {k: (x["diff"], x["ci95"]) for k, x in v["paired_fresh"]["metrics"].items()
                     if k in ("intent_acc", "macro_f1", "action_acc", "noisy_gap", "top70_error_ratio")})


if __name__ == "__main__":
    main()
