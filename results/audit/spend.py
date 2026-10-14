"""Recompute API spend from the ledger usage counts at list price (own arithmetic)."""
import json, collections, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
P = {"in": 0.10, "out": 0.50, "cache_read": 0.01, "cache_write": 0.125}  # USD / M tokens, both models (llm_common.PRICES)
L = [json.loads(l) for l in open(ROOT / "results/raw/llm-ledger/ledger.jsonl")]
agg = collections.defaultdict(lambda: collections.Counter())
mism = 0
for r in L:
    u = r["usage"] or {}
    if r["provider"] == "anthropic":   # Anthropic usage: input_tokens excludes cache reads and writes
        cr, cw = u.get("cache_read") or 0, u.get("cache_write") or 0
        pt, ct = (u.get("input_tokens") or 0) + cr + cw, u.get("output_tokens") or 0
    else:                              # OpenAI-style usage: prompt_tokens includes cached and written tokens
        pt, ct = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
        cr, cw = u.get("cached_tokens") or 0, u.get("cache_write_tokens") or 0
    unc = (pt * P["in"] + ct * P["out"]) / 1e6
    cac = ((pt - cr - cw) * P["in"] + cr * P["cache_read"] + cw * P["cache_write"] + ct * P["out"]) / 1e6
    k = (r["provider"], r["tag"])
    a = agg[k]
    a["attempts"] += 1; a["ok"] += r["status"] == "ok"
    a["uncached"] += unc; a["cached"] += cac; a["reported"] += r.get("cost_reported") or 0
    a["in_tok"] += pt; a["out_tok"] += ct
    if abs(unc - (r.get("cost_list_uncached") or 0)) > 1e-9 or abs(cac - (r.get("cost_list_actual") or 0)) > 1e-9:
        mism += 1
out = {"per_tag": {f"{p}/{t}": {k: round(v, 6) if isinstance(v, float) else v for k, v in a.items()} for (p, t), a in sorted(agg.items())}}
prov = collections.defaultdict(collections.Counter)
for (p, t), a in agg.items():
    for k, v in a.items():
        prov[p][k] += v
out["per_provider"] = {p: {k: round(v, 4) if isinstance(v, float) else v for k, v in a.items()} for p, a in prov.items()}
out["rows_where_ledger_cost_differs_from_recomputed"] = mism
out["status_counts"] = dict(collections.Counter(r["status"] for r in L))
out["models_requested"] = sorted({r["model"] for r in L})
out["models_returned"] = sorted({str(r.get("model_returned")) for r in L})
out["providers_returned"] = sorted({str(r.get("provider_returned")) for r in L})
out["first_ts"] = min(r["ts"] for r in L); out["last_ts"] = max(r["ts"] for r in L)
json.dump(out, open(ROOT / "results/audit/spend.json", "w"), indent=1)
print(json.dumps(out, indent=1))
