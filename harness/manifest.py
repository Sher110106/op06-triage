"""Rebuild results/manifest.json from what is on disk.

  python -m harness.manifest

Every folder under results/raw/ is one run and one entry in `runs`, except results/raw/llm-ledger/, the API call
ledger that every LLM run appends to, which is listed under `ledger`. Kinds of run:

  graded      predictions.jsonl with the harness report (report.json) and the grader's own (report_official.json):
              dev runs, ablations, taxonomy columns, the LLM-test systems, and chaos runs
  ungraded    predictions with no grader report: Linux load tests, the hostile-in-context probe
  llm_calls   raw API call logs (raw_calls.jsonl), with token counts and cost at list price
  encoder     the fine-tuned encoder's probability files (one folder, no single main file)
  incomplete  a folder with neither predictions nor call logs (a run that failed before writing them); listed with
              status "failed" so failed runs stay in the record

Fields of every run: run_id, kind, seed, model, status, input_sha256, input_files, raw_path (the run's main raw file,
or its folder when it has none), created (where known). `input_sha256` is the SHA-256 of the files in
`input_files` concatenated in that order: the data a run read its rows from. A run that used only some rows (chaos
runs, smoke tests) says so in `rows`. Runs graded with rows left out (chaos runs) report the harness's numbers on
the rows sent, because the grader counts unsent rows as wrong.

`claimed` comes from the run named in CLAIM_RUN and uses the submission's OP-06 claim keys.
The build stops if a raw path is missing, is a symlink, leaves the repo, or a folder in results/raw/ is not listed.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from triage.data import DATA, ROOT

RAW = ROOT / "results" / "raw"
CLAIM_RUN = "dev-classical-v0"
LEDGER_DIR = RAW / "llm-ledger"
ALL_ROWS = ["data/op06/dev.jsonl", "data/op06/dev_noisy.jsonl", "data/op06/hostile.jsonl"]

# Facts that a run's own files do not hold, with where each comes from.
KNOWN = {
    "dev-baseline": {
        "model": "published-baseline (data/op06/baseline.py)", "seed": None,
        "seed_note": "the baseline has no random step",
        "source": "EXPERIMENT_LOG P0-2: data/op06/baseline.py rerun on 14 Oct and graded"},
    "dev-pipeline-llm-replay": {
        "bundle_sha256": "c007c7a3769a48e845f21d079230d62b46d860bebe6ebcfdb47863cab195357c",
        "llm": "openai/gpt-6-luna-20260922 (recorded answers replayed, no API call)",
        "source": "meta.json has no bundle hash; results/raw/ablation-ref-llm, run with this bundle, reproduces "
                  "these predictions byte for byte"},
    "llm-tail-A_luna_tail": {
        "bundle_sha256": "c007c7a3769a48e845f21d079230d62b46d860bebe6ebcfdb47863cab195357c",
        "system": "classical", "llm": "openai/gpt-6-luna-20260922 (answers from results/raw/llm-tail-luna_V3_none)",
        "source": "experiments/llm_tail/analyse.py: dev-classical-v0 predictions with recorded LLM answers"},
    "llm-tail-B_haiku_intent": {
        "bundle_sha256": "c007c7a3769a48e845f21d079230d62b46d860bebe6ebcfdb47863cab195357c",
        "system": "classical", "llm": "claude-haiku-5-5 (answers from results/raw/llm-tail-haiku_V3_low)",
        "source": "experiments/llm_tail/analyse.py: dev-classical-v0 predictions with recorded LLM answers"},
    "llm-tail-C_both": {
        "bundle_sha256": "c007c7a3769a48e845f21d079230d62b46d860bebe6ebcfdb47863cab195357c",
        "system": "classical",
        "llm": "openai/gpt-6-luna-20260922 + claude-haiku-5-5 (answers from results/raw/llm-tail-luna_V3_none and "
               "llm-tail-haiku_V3_low)",
        "source": "experiments/llm_tail/analyse.py: dev-classical-v0 predictions with recorded LLM answers"},
}


def evidence(run_id: str) -> str:
    if run_id.startswith(("ablation-", "chaos-", "taxonomy-")):
        return "exploratory, unaudited"
    if run_id.startswith(("llm-tail-", "llm-confirm-", "encoder-")):
        return "pre-registered test"
    if run_id.startswith("linux-"):
        return "development measurement, Linux box"
    return "development measurement"


# ----------------------------------------------------------------------------- helpers

def sha256_files(files: list[str]) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update((ROOT / f).read_bytes())
    return h.hexdigest()


def file_sha(f: str) -> str:
    return hashlib.sha256((ROOT / f).read_bytes()).hexdigest()


def read_json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def inputs_of(meta: dict, default: list[str]) -> tuple[list[str], bool | None]:
    """The data files a run read rows from, and whether their SHA-256 still equals what the run recorded."""
    inp = meta.get("inputs")
    if isinstance(inp, dict) and isinstance(inp.get("data"), dict):
        inp = inp["data"]
    if isinstance(inp, dict) and inp and all(isinstance(v, dict) and "file" in v for v in inp.values()):
        files = [v["file"] for v in inp.values()]
        return files, all(file_sha(v["file"]) == v["sha256"] for v in inp.values() if v.get("sha256"))
    if meta.get("input_files"):
        return list(meta["input_files"]), None
    return default, None


def bundle_seeds() -> dict[str, int]:
    """bundle SHA-256 -> training seed, from every run record that holds both."""
    out = {}
    for m in RAW.glob("*/meta.json"):
        d = read_json(m)
        if d.get("bundle_sha256") and isinstance(d.get("bundle_meta"), dict) and "seed" in d["bundle_meta"]:
            out[d["bundle_sha256"]] = d["bundle_meta"]["seed"]
    return out


def n_lines(p: Path) -> int:
    with open(p, "rb") as fh:
        return sum(1 for l in fh if l.strip())


# ----------------------------------------------------------------------------- runs

def llm_run(d: Path, ledger: Counter) -> dict:
    meta = read_json(d / "meta.json")
    f = d / "raw_calls.jsonl"
    recs = read_jsonl(f)
    st = Counter(r.get("status") for r in recs)
    cfg = recs[0].get("cfg") or {}
    tag = recs[0].get("tag")
    files, _ = inputs_of(meta, ALL_ROWS[:2])
    u = [r.get("usage") or {} for r in recs]
    return {
        "run_id": d.name, "kind": "llm_calls", "seed": None,
        "seed_note": "provider-side sampling; no seed can be set",
        "model": recs[0].get("model"), "status": "completed" if st.get("ok") == len(recs) else "partial",
        "input_sha256": sha256_files(files), "input_files": files, "raw_path": str(f.relative_to(ROOT)),
        "created": meta.get("created"), "finished": meta.get("finished"), "evidence": evidence(d.name),
        "rows": len(recs), "calls_ok": st.get("ok", 0), "ledger_attempts": ledger.get(tag, 0),
        "config": {k: cfg.get(k) for k in ("effort", "reasoning", "provider_order", "no_fallbacks", "prompt_version")}
                  | {"variant": recs[0].get("variant")},
        "models_returned": sorted({str(r.get("model_returned")) for r in recs}),
        "providers_returned": sorted({str(r.get("provider_returned")) for r in recs}),
        "parse_status": dict(Counter(str(r.get("parse_status")) for r in recs)),
        "input_tokens": sum(x.get("input_total", 0) for x in u),
        "cached_input_tokens": sum(x.get("cached_tokens") or x.get("cache_read") or 0 for x in u),
        "output_tokens": sum(x.get("output_tokens", x.get("completion_tokens", 0)) for x in u),
        "api_cost_usd_list_with_cache": round(sum(r.get("cost_list_actual") or 0 for r in recs), 4),
        "api_cost_usd_list_uncached": round(sum(r.get("cost_list_uncached") or 0 for r in recs), 4),
        "api_cost_usd_reported_by_provider": (round(sum(r["cost_reported"] for r in recs if r.get("cost_reported") is not None), 4)
                                              if any(r.get("cost_reported") is not None for r in recs) else None),
        "script": meta.get("script"),
    }


def encoder_run(d: Path) -> dict:
    meta = read_json(d / "meta.json")
    files, _ = inputs_of(meta, [])
    return {
        "run_id": d.name, "kind": "encoder", "seed": meta.get("seed"), "seed_note": meta.get("seed_note"),
        "model": f"{meta.get('model')} (fine-tuned on train fold A, tag {meta.get('tag')})",
        "status": "completed", "input_sha256": sha256_files(files), "input_files": files,
        "raw_path": str(d.relative_to(ROOT)), "created": meta.get("created"), "created_note": meta.get("created_note"),
        "evidence": evidence(d.name), "files": len([p for p in d.iterdir() if p.name != "meta.json"]),
        "config": meta.get("config"), "analysis": meta.get("analysis"), "api_cost_usd": 0.0,
    }


def system_run(d: Path, seeds: dict) -> dict:
    """A run that predicts on rows: graded or not."""
    meta = read_json(d / "meta.json")
    bench = read_json(d / "latency_bench.json")
    rep = read_json(d / "report.json")
    known = KNOWN.get(d.name, {})
    health = bench.get("server_health") or {}
    bundle = meta.get("bundle_sha256") or health.get("bundle_sha256") or known.get("bundle_sha256")
    system = meta.get("system") or known.get("system") or ("service-http" if bench else None)
    model = known.get("model") or (f"{(system or 'pipeline').split(' ')[0]}@{bundle[:12]}" if bundle else system)
    llm = known.get("llm")
    llm_meta = meta.get("llm") if isinstance(meta.get("llm"), dict) else (health.get("llm") or {})
    fake = meta.get("fake_upstream")
    if llm is None and llm_meta:
        if llm_meta.get("mode") == "replay":
            llm = "openai/gpt-6-luna-20260922 (recorded answers replayed, no API call)"
        elif llm_meta.get("enabled") and fake:
            llm = f"fake upstream in mode {fake.get('mode')!r} (no API call)"
        elif llm_meta.get("enabled") and llm_meta.get("base_url_host") == "127.0.0.1":
            llm = "LLM endpoint on 127.0.0.1 (fake upstream or a closed port; no API call)"
        elif llm_meta.get("enabled"):
            llm = llm_meta.get("model")
    if llm is None and meta.get("llm_answers"):
        llm = "openai/gpt-6-luna-20260922 (recorded answers replayed, no API call)"
    if llm:
        model = f"{model} + {llm}"
    seed = known["seed"] if "seed" in known else (meta.get("bundle_meta") or {}).get("seed", seeds.get(bundle))
    # chaos runs send dev rows, then dev_noisy rows, in order; the others predict on all 2,309 rows
    files, inputs_match = inputs_of(meta, ALL_ROWS[:2] if d.name.startswith("chaos-") else ALL_ROWS)
    pred = d / "predictions.jsonl"
    real_llm = bool(llm_meta.get("enabled")) and not fake and llm_meta.get("base_url_host") not in (None, "127.0.0.1")
    out = {
        "run_id": d.name, "kind": "graded" if rep else "ungraded", "seed": seed,
        **({"seed_note": known["seed_note"]} if "seed_note" in known else {}),
        "model": model, "bundle_sha256": bundle, "status": "completed",
        "input_sha256": sha256_files(files), "input_files": files, "raw_path": str(pred.relative_to(ROOT)),
        "created": meta.get("created") or bench.get("created"), "evidence": evidence(d.name),
        "rows": n_lines(pred),
    }
    if files == ALL_ROWS and out["rows"] != sum(n_lines(ROOT / f) for f in ALL_ROWS):
        out["status"] = "partial"                   # fewer answers than the 2,309 rows it was given
    if inputs_match is not None:
        out["input_files_match_run_record"] = inputs_match
    if isinstance(meta.get("inputs"), dict) and meta["inputs"].get("inputs_sha256"):
        out["input_subset_sha256"] = meta["inputs"]["inputs_sha256"]
        out["input_subset"] = meta["inputs"].get("recipe")
    if known.get("source"):
        out["facts_source"] = known["source"]
    for k in ("status", "what_changed", "changed"):
        if isinstance(meta.get(k), str):
            out["run_note" if k == "status" else "changed"] = meta[k]
    if bench:
        out["load"] = {k: bench.get(k) for k in ("concurrency", "sample_count", "errors", "non_200", "schema_invalid",
                                                 "latency_ms", "throughput_rps")}
    # cost: only real provider calls count; replays and fake upstreams cost nothing
    if real_llm and bench.get("llm_calls") is not None:
        out["llm_calls"] = bench["llm_calls"]
        out["api_cost_usd"] = bench.get("llm_cost_usd")
        out["api_cost_note"] = ("list price from the token counts the provider returned (GET /metrics before and "
                                "after the run); per-call token counts were not kept")
        before = ((bench.get("server_metrics_before") or {}).get("llm") or {})
        if before.get("calls"):
            out["llm_calls_before_window"] = before["calls"]
            out["api_cost_usd_before_window"] = before.get("cost_usd")
    else:
        out["api_cost_usd"] = 0.0
    if rep:
        out["grader_version"] = rep.get("grader_version")
        ex = rep.get("extras") or {}
        if ex.get("excluded"):
            out["metrics_scope"] = (f"harness numbers on the {ex.get('rows')} labelled rows sent; the grader's own "
                                    f"report counts the {ex['excluded']} unsent rows as wrong")
            out["metrics"] = ex.get("point")
            out["bars_pass"] = None
        else:
            out["metrics"] = rep["official"]
            out["bars_pass"] = rep["bars"]
    return out


def incomplete_run(d: Path) -> dict:
    """A folder with no predictions and no call log: a run that stopped before writing its outputs."""
    meta = read_json(d / "meta.json")
    files, _ = inputs_of(meta, ALL_ROWS)
    bundle = meta.get("bundle_sha256")
    return {"run_id": d.name, "kind": "incomplete", "seed": (meta.get("bundle_meta") or {}).get("seed"),
            "model": f"{meta.get('system', 'unknown')}@{bundle[:12]}" if bundle else meta.get("system"),
            "status": "failed", "input_sha256": sha256_files(files), "input_files": files,
            "raw_path": str(d.relative_to(ROOT)), "created": meta.get("created"), "evidence": evidence(d.name),
            "files": sorted(p.name for p in d.iterdir()), "api_cost_usd": None,
            "note": "no predictions.jsonl or raw_calls.jsonl in this folder"}


def ledger_entry() -> tuple[dict, Counter]:
    f = LEDGER_DIR / "ledger.jsonl"
    recs = read_jsonl(f)
    tags = Counter(r["tag"] for r in recs)
    by_provider = {}
    for r in recs:
        p = by_provider.setdefault(r["provider"], {"attempts": 0, "ok": 0, "list_with_cache": 0.0, "list_uncached": 0.0,
                                                    "reported": 0.0})
        p["attempts"] += 1
        p["ok"] += r.get("status") == "ok"
        p["list_with_cache"] += r.get("cost_list_actual") or 0
        p["list_uncached"] += r.get("cost_list_uncached") or 0
        p["reported"] += r.get("cost_reported") or 0
    for p in by_provider.values():
        for k in ("list_with_cache", "list_uncached", "reported"):
            p[k] = round(p[k], 4)
    meta = read_json(LEDGER_DIR / "meta.json")
    return {"raw_path": str(f.relative_to(ROOT)), "keycheck": str((LEDGER_DIR / "keycheck.jsonl").relative_to(ROOT)),
            "run_summaries": str((LEDGER_DIR / "run_summaries.jsonl").relative_to(ROOT)),
            "about": meta.get("about"), "attempts": len(recs), "first": min(r["ts"] for r in recs),
            "last": max(r["ts"] for r in recs), "by_provider_usd": by_provider,
            "prior_spend_outside_repo": meta.get("prior_spend_not_in_this_ledger")}, tags


def check(runs: list[dict], ledger: dict) -> dict:
    """Every raw path exists inside the repo with no symlink on the way; every folder is listed once."""
    ids = [r["run_id"] for r in runs]
    assert len(ids) == len(set(ids)), "duplicate run_id"
    paths = [r["raw_path"] for r in runs] + [ledger["raw_path"], ledger["keycheck"], ledger["run_summaries"]]
    paths += sorted({f for r in runs for f in r["input_files"]})
    root = ROOT.resolve()
    for p in paths:
        assert not Path(p).is_absolute() and ".." not in Path(p).parts, f"unsafe path {p}"
        full = ROOT / p
        assert full.exists(), f"missing {p}"
        for q in [full, *full.parents]:
            if q == ROOT:
                break
            assert not q.is_symlink(), f"symlink on the way to {p}: {q}"
        assert full.resolve().is_relative_to(root), f"{p} leaves the repo"
    folders = sorted(p.name for p in RAW.iterdir() if p.is_dir())
    listed = set(ids) | {LEDGER_DIR.name}
    missing = [f for f in folders if f not in listed]
    assert not missing, f"folders in results/raw/ not in the manifest: {missing}"
    return {"raw_paths_checked": len(paths), "all_exist_inside_repo_no_symlinks": True,
            "folders_in_results_raw": len(folders), "all_folders_listed": True}


def main():
    ledger, tags = ledger_entry()
    seeds = bundle_seeds()
    runs = []
    for d in sorted(p for p in RAW.iterdir() if p.is_dir() and p != LEDGER_DIR):
        if (d / "raw_calls.jsonl").exists():
            runs.append(llm_run(d, tags))
        elif d.name.startswith("encoder-"):
            runs.append(encoder_run(d))
        elif (d / "predictions.jsonl").exists():
            runs.append(system_run(d, seeds))
        else:
            runs.append(incomplete_run(d))
    claim = next(r for r in runs if r["run_id"] == CLAIM_RUN)["metrics"]
    claimed = {"intent_accuracy": claim["intent_acc"], "macro_f1": claim["macro_f1"], "action_accuracy": claim["action_acc"],
               "noisy_accuracy": claim["action_acc_noisy"], "schema_valid_pct": claim["schema_valid_pct"],
               "top70_error_ratio": claim["top70_error_ratio"], "cost_per_message_usd": None, "latency_p95_s": None}
    llm = [r for r in runs if r["kind"] == "llm_calls"]
    paid = [r for r in runs if r.get("llm_calls")]
    spend = {
        "llm_call_runs_list_with_cache": round(sum(r["api_cost_usd_list_with_cache"] for r in llm), 4),
        "llm_call_runs_list_uncached": round(sum(r["api_cost_usd_list_uncached"] for r in llm), 4),
        "llm_call_runs_reported_by_openrouter": round(sum(r["api_cost_usd_reported_by_provider"] or 0 for r in llm), 4),
        "service_runs_list": round(sum((r.get("api_cost_usd") or 0) + (r.get("api_cost_usd_before_window") or 0)
                                       for r in paid), 4),
        "service_runs": [r["run_id"] for r in paid],
        "note": "US dollars at the list prices in experiments/llm_tail/llm_common.PRICES and triage/llm_action.py. "
                "Not included: the 13 Oct probe (outside this repo; see ledger.prior_spend_outside_repo).",
    }
    manifest = {"problem": "OP-06", "claimed": claimed, "claimed_from_run": CLAIM_RUN,
                "note": "Development measurements on the published dev split. cost and latency stay null until the "
                        "Phase 3 load test measures them.",
                "input_sha256_definition": "SHA-256 of the run's input_files concatenated in the order listed",
                "spend_usd": spend, "ledger": ledger, "runs": runs}
    manifest["checks"] = check(runs, ledger)
    (ROOT / "results" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"runs": len(runs), "kinds": dict(Counter(r["kind"] for r in runs)), "claimed": claimed,
                      "spend_usd": {k: v for k, v in spend.items() if k != "note"}, "checks": manifest["checks"]}))


if __name__ == "__main__":
    main()
