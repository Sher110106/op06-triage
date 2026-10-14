"""Rebuild results/manifest.json from what is on disk.

  python -m harness.manifest

Every graded run under results/raw/ becomes one entry; so does every LLM run under experiments/*/runs/
(including smoke tests, which count towards spend). `claimed` comes from the run named in CLAIM_RUN and
uses the submission's OP-06 claim keys. `input_sha256` is the SHA-256 of dev.jsonl, dev_noisy.jsonl and
hostile.jsonl concatenated in that order (the exact bytes a run predicts on).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from triage.data import DATA, ROOT

RAW = ROOT / "results" / "raw"
CLAIM_RUN = "dev-classical-v0"


def inputs_sha() -> str:
    h = hashlib.sha256()
    for f in ("dev.jsonl", "dev_noisy.jsonl", "hostile.jsonl"):
        h.update((DATA / f).read_bytes())
    return h.hexdigest()


def graded_runs(sha):
    runs = []
    for d in sorted(p for p in RAW.iterdir() if p.is_dir()):
        rep = d / "report.json"
        if not rep.exists():
            continue
        r = json.loads(rep.read_text())
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        model = meta.get("system") or ("published-baseline" if d.name == "dev-baseline" else d.name)
        if meta.get("bundle_sha256"):
            model += f"@{meta['bundle_sha256'][:12]}"
        runs.append({"run_id": d.name, "seed": (meta.get("bundle_meta") or {}).get("seed", 0), "model": model,
                     "status": "completed", "input_sha256": sha, "raw_path": f"results/raw/{d.name}/predictions.jsonl",
                     "created": meta.get("created"), "grader_version": r.get("grader_version"),
                     "metrics": r["official"], "bars_pass": r["bars"], "api_cost_usd": 0.0})
    return runs


def llm_runs(sha):
    runs = []
    for f in sorted(ROOT.glob("experiments/*/runs/*.jsonl")):
        recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not recs:
            continue
        st = Counter(r.get("status") for r in recs)
        cfg = recs[0].get("cfg") or {}
        runs.append({"run_id": f"{f.parent.parent.name}-{f.stem}", "seed": None, "model": recs[0].get("model"),
                     "status": "completed" if st.get("ok") == len(recs) else "partial",
                     "input_sha256": sha, "raw_path": str(f.relative_to(ROOT)), "rows": len(recs), "calls_ok": st.get("ok", 0),
                     "config": {k: cfg.get(k) for k in ("effort", "reasoning", "provider_order", "no_fallbacks", "prompt_version")}
                               | {"variant": recs[0].get("variant")},
                     "models_returned": sorted({str(r.get("model_returned")) for r in recs}),
                     "providers_returned": sorted({str(r.get("provider_returned")) for r in recs}),
                     "input_tokens": sum((r.get("usage") or {}).get("input_total", 0) for r in recs),
                     "output_tokens": sum((r.get("usage") or {}).get("output_tokens", (r.get("usage") or {}).get("completion_tokens", 0)) for r in recs),
                     "api_cost_usd_list_with_cache": round(sum(r.get("cost_list_actual") or 0 for r in recs), 4),
                     "api_cost_usd_list_uncached": round(sum(r.get("cost_list_uncached") or 0 for r in recs), 4)})
    return runs


def main():
    sha = inputs_sha()
    runs = graded_runs(sha) + llm_runs(sha)
    claim = next(r for r in runs if r["run_id"] == CLAIM_RUN)["metrics"]
    claimed = {"intent_accuracy": claim["intent_acc"], "macro_f1": claim["macro_f1"], "action_accuracy": claim["action_acc"],
               "noisy_accuracy": claim["action_acc_noisy"], "schema_valid_pct": claim["schema_valid_pct"],
               "top70_error_ratio": claim["top70_error_ratio"], "cost_per_message_usd": None, "latency_p95_s": None}
    manifest = {"problem": "OP-06", "claimed": claimed, "claimed_from_run": CLAIM_RUN,
                "note": "Development measurements on the published dev split. cost and latency stay null until the "
                        "Phase 3 load test measures them.",
                "runs": runs}
    (ROOT / "results" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"runs": len(runs), "claimed": claimed}))


if __name__ == "__main__":
    main()
