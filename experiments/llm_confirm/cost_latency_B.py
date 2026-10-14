"""Add the "action_step_cost_latency" block to results_B.json (run after analyse_B.py). No API calls.

Reconstructed on 14 Oct 2026. The committed results_B.json carries this block, but analyse_B.py does not write it
and no script in the repo did: it was added by a step that was not saved. This script rebuilds it from the raw
files, and its output equals the committed block in all 11 numbers. The method is inferred from those numbers, not
recorded, so treat it as a reconstruction:

- share: fold-B rows whose out-of-fold CPU confidence is below the gate T20 (the rows the Luna action step calls).
- per_call_*: mean list-price cost per Luna call over all fold-B calls, with cache and uncached.
- per1000_*: share x per-call cost x 1,000; per1000_at_dev_share20_uncached uses the 20% share fixed on dev.
- llm_p50 / llm_p95: quantiles of the recorded Luna call latency on the gated rows (numpy, linear).
- overall_*: per-row latency = 15 ms of CPU work (EXPERIMENT_LOG R7: in-process p95 14.6 ms) plus the recorded
  Luna latency on gated rows. An offline estimate from single calls at concurrency 8, not a load test.

  python experiments/llm_confirm/cost_latency_B.py [--luna results/raw/llm-confirm-luna_B/raw_calls.jsonl] [--check]

--check prints the rebuilt block next to the one in results_B.json and writes nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
T20 = 0.642841          # the action-step gate, as in analyse_B.py
DEV_SHARE = 0.20        # the share fixed on dev before the fold-B test
CPU_S = 0.015           # CPU time per row, seconds


def block(luna_path: Path) -> dict:
    cpu = {json.loads(l)["id"]: json.loads(l) for l in (HERE / "cpu_B_predictions.jsonl").read_text().splitlines()}
    recs = {r["id"]: r for r in map(json.loads, Path(luna_path).read_text(encoding="utf-8").splitlines())}
    gated = [rid for rid, p in cpu.items() if p["confidence"] < T20]
    share = round(len(gated) / len(cpu), 4)
    per_cache = round(float(np.mean([r["cost_list_actual"] for r in recs.values()])), 6)
    per_unc = round(float(np.mean([r["cost_list_uncached"] for r in recs.values()])), 6)
    llm = np.array([recs[rid]["latency_s"] for rid in gated])
    overall = np.array([CPU_S + (recs[rid]["latency_s"] if p["confidence"] < T20 else 0.0) for rid, p in cpu.items()])
    return {"share": share, "per_call_cache": per_cache, "per_call_uncached": per_unc,
            "per1000_cache": round(share * per_cache * 1000, 4), "per1000_uncached": round(share * per_unc * 1000, 4),
            "per1000_at_dev_share20_uncached": round(DEV_SHARE * per_unc * 1000, 4),
            "llm_p50": round(float(np.quantile(llm, 0.5)), 2), "llm_p95": round(float(np.quantile(llm, 0.95)), 2),
            "overall_p50": round(float(np.quantile(overall, 0.5)), 3),
            "overall_p95": round(float(np.quantile(overall, 0.95)), 2),
            "overall_p99": round(float(np.quantile(overall, 0.99)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--luna", default=str(ROOT / "results/raw/llm-confirm-luna_B/raw_calls.jsonl"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    path = HERE / "results_B.json"
    res = json.loads(path.read_text())
    new = block(Path(a.luna))
    if a.check:
        old = res.get("action_step_cost_latency")
        print(json.dumps({"rebuilt": new, "in_results_B": old, "equal": new == old}, indent=1))
        return
    res.pop("action_step_cost_latency", None)
    res["action_step_cost_latency"] = new
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps(new))


if __name__ == "__main__":
    main()
