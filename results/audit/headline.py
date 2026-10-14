"""Compare own-scorer headline numbers with the official report and manifest claims; recompute latency and cost
from results/latency_bench.json per-request records. Writes results/audit/headline.json."""
import json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from score import score

mine = score(ROOT / "results/predictions.jsonl")
off = json.load(open(ROOT / "results/raw/dev-http-llm/report_official.json"))
man = json.load(open(ROOT / "results/manifest.json"))["claimed"]
pairs = {
    "intent_accuracy": (mine["intent_acc_all"], off["intent"]["accuracy_all"], man.get("intent_accuracy")),
    "macro_f1": (mine["macro_f1_all"], off["intent"]["macro_f1_all"], man.get("macro_f1")),
    "action_accuracy": (mine["action_acc_all"], off["action"]["accuracy_all"], man.get("action_accuracy")),
    "action_clean": (mine["action_acc_clean"], off["action"]["accuracy_clean"], None),
    "noisy_accuracy(action_noisy)": (mine["action_acc_noisy"], off["action"]["accuracy_noisy"], man.get("noisy_accuracy")),
    "noisy_gap": (mine["noisy_gap"], off["action"]["noisy_gap"], None),
    "top70_error_ratio": (mine["top70_error_ratio"], off["confidence"]["top70_error_ratio"], man.get("top70_error_ratio")),
    "schema_valid_pct": (mine["schema_valid_pct"], 100 * off["schema"]["valid_rate_all_inputs"], man.get("schema_valid_pct")),
    "policy_violation": (mine["policy_violation"], off["policy"]["violation_rate"], None),
}
rows = {k: {"own": round(a, 4), "official_report": b, "manifest_claimed": c,
            "own_eq_official": abs(round(a, 4) - b) < 1e-9,
            "claimed_eq_own": None if c is None else abs(round(a, 4) - c) < 5e-4} for k, (a, b, c) in pairs.items()}
lb = json.load(open(ROOT / "results/latency_bench.json"))
ds = sorted(r["duration_ms"] for r in lb["requests"])
n = len(ds)
def nearest(p): return ds[max(0, math.ceil(p * n) - 1)]
def linear(p):
    k = (n - 1) * p; f = math.floor(k); c = min(f + 1, n - 1); return ds[f] + (ds[c] - ds[f]) * (k - f)
lat = {"n": n, "concurrency": lb["concurrency"], "status_200": sum(r["status"] == 200 for r in lb["requests"]),
       "p50_ms_linear": round(linear(0.5), 3), "p95_ms_linear": round(linear(0.95), 3), "p95_ms_nearest_rank": nearest(0.95),
       "p99_ms_linear": round(linear(0.99), 3), "reported_p95_ms": lb["latency_ms"]["p95"],
       "requests_over_500ms": sum(d > 500 for d in ds), "llm_calls_reported": lb["llm_calls"],
       "llm_cost_usd_reported_by_service": lb["llm_cost_usd"],
       "cost_per_message_usd": lb["llm_cost_usd"] / n, "cost_per_1000_usd": 1000 * lb["llm_cost_usd"] / n,
       "note": "per-request records carry no cost; cost is the service's own /metrics delta at list price with caching"}
out = {"headline": rows, "manifest_claimed_from_run": json.load(open(ROOT / "results/manifest.json")).get("claimed_from_run"),
       "latency_cost": lat}
(ROOT / "results/audit/headline.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
