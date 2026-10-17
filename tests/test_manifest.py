"""harness/manifest.py: the claim comes from CLAIM_RUN, with p95 latency and cost per message from that run's
latency_bench.json, and results/manifest.json holds what the code builds now. Reads results/; writes nothing."""
from __future__ import annotations

import json

import pytest

from harness import manifest as M
from triage.data import ROOT

BENCH = M.RAW / M.CLAIM_RUN / "latency_bench.json"
METRICS = {"intent_acc": 0.7, "macro_f1": 0.69, "action_acc": 0.8, "action_acc_noisy": 0.78, "schema_valid_pct": 100.0,
           "top70_error_ratio": 0.4}


@pytest.fixture(scope="module")
def built():
    """(run entry, bench, claimed, claimed_notes) for CLAIM_RUN, as main() builds them."""
    if not BENCH.exists():
        pytest.skip(f"{BENCH.relative_to(ROOT)} is missing")
    run = M.system_run(M.RAW / M.CLAIM_RUN, M.bundle_seeds())
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    return run, bench, *M.claims(run, bench)


def test_claim_run_is_the_http_run_with_the_llm_step(built):
    run, bench, _, _ = built
    assert M.CLAIM_RUN == "dev-http-llm-20261017"
    assert run["kind"] == "graded" and run["status"] == "completed" and run["rows"] == 2309
    assert bench["sample_count"] == 2309 and bench["concurrency"] == 8
    assert bench["errors"] == bench["non_200"] == bench["schema_invalid"] == bench["id_mismatch"] == 0
    assert bench["server_health"]["llm"]["enabled"] is True and bench["llm_calls"] > 0


def test_latency_and_cost_claims_match_the_bench(built):
    _, bench, claimed, notes = built
    assert claimed["latency_p95_s"] is not None and claimed["latency_p95_s"] > 0
    assert claimed["latency_p95_s"] == round(bench["latency_ms"]["p95"] / 1000, 3)
    key = ("cost_per_message_uncached_usd" if bench.get("cost_per_message_uncached_usd") is not None
           else "cost_per_message_usd")
    assert claimed["cost_per_message_usd"] is not None and claimed["cost_per_message_usd"] > 0
    assert claimed["cost_per_message_usd"] == bench[key]
    assert notes["cost_source"] == f"latency_bench.json {key}"
    assert notes["cost_per_message_usd_cached_rate"] == bench["cost_per_message_usd"]
    assert notes["cost_per_message_usd_uncached_rate"] == bench.get("cost_per_message_uncached_usd")


def test_grader_claims_match_the_report(built):
    _, _, claimed, notes = built
    rep = json.loads((M.RAW / M.CLAIM_RUN / "report.json").read_text(encoding="utf-8"))["official"]
    pairs = {"intent_accuracy": "intent_acc", "macro_f1": "macro_f1", "action_accuracy": "action_acc",
             "noisy_accuracy": "action_acc_noisy", "schema_valid_pct": "schema_valid_pct",
             "top70_error_ratio": "top70_error_ratio"}
    assert {k: claimed[k] for k in pairs} == {k: rep[v] for k, v in pairs.items()}
    assert all(isinstance(notes[k], str) and notes[k] for k in claimed)      # a plain note for every claim key
    assert notes["noisy_accuracy"] == "action accuracy on dev_noisy (300 rows)"
    assert "concurrency 8" in notes["latency_p95_s"] and "LLM step on" in notes["latency_p95_s"]


def test_cost_claim_prefers_the_uncached_figure():
    bench = {"latency_ms": {"p95": 1234.5678}, "cost_per_message_usd": 1e-05, "cost_per_message_uncached_usd": 3e-05,
             "sample_count": 10, "concurrency": 8, "server_health": {"llm": {"enabled": True}}}
    claimed, notes = M.claims({"metrics": METRICS}, bench)
    assert claimed["cost_per_message_usd"] == 3e-05 and claimed["latency_p95_s"] == 1.235
    assert notes["cost_source"] == "latency_bench.json cost_per_message_uncached_usd"
    assert "uncached rates" in notes["cost_per_message_usd"] and "1e-05" in notes["cost_per_message_usd"]
    del bench["cost_per_message_uncached_usd"]
    claimed, notes = M.claims({"metrics": METRICS}, bench)
    assert claimed["cost_per_message_usd"] == 1e-05
    assert notes["cost_source"] == "latency_bench.json cost_per_message_usd"
    assert "WITH cache discounts" in notes["cost_per_message_usd"]
    assert notes["cost_per_message_usd_uncached_rate"] is None


def test_manifest_on_disk_holds_the_claim(built):
    """results/manifest.json was rebuilt after the last change to the claim code (python -m harness.manifest)."""
    _, _, claimed, notes = built
    m = json.loads((ROOT / "results" / "manifest.json").read_text(encoding="utf-8"))
    assert m["claimed_from_run"] == M.CLAIM_RUN
    assert m["claimed"] == claimed and m["claimed_notes"] == notes
    assert "stay null" not in m["note"]


def test_canonical_predictions_are_the_current_claim_run(built):
    """A reviewer must receive exactly the predictions described by the manifest."""
    canonical = ROOT / "results" / "predictions.jsonl"
    assert canonical.read_bytes() == (M.RAW / M.CLAIM_RUN / "predictions.jsonl").read_bytes()
    split_bytes = b"".join((ROOT / "results" / name).read_bytes() for name in
                           ("dev_predictions.jsonl", "noisy_predictions.jsonl", "hostile_results.jsonl"))
    assert canonical.read_bytes() == split_bytes
    assert json.loads((ROOT / "results" / "latency_bench.json").read_text()) == json.loads(BENCH.read_text())


def test_original_submission_remains_inspectable():
    archive = ROOT / "results" / "history" / "2026-10-14"
    original = json.loads((archive / "manifest.json").read_text())
    assert original["claimed_from_run"] == "dev-http-llm"
    assert (archive / "predictions.jsonl").read_bytes() == (M.RAW / "dev-http-llm" / "predictions.jsonl").read_bytes()


def test_refinement_spend_is_not_double_counted():
    manifest = json.loads((ROOT / "results" / "manifest.json").read_text())
    extra = manifest["spend_usd"]["refinement_20261017"]
    ledger = json.loads((ROOT / extra["source"]).read_text())
    assert extra["recorded_total_with_cache"] == ledger["recorded_reported_or_cached_list_usd"]
    assert extra["already_in_service_runs_list"] + extra["additional_to_service_runs_list"] == pytest.approx(
        extra["recorded_total_with_cache"], abs=1e-8)


def test_model_exploration_index_is_additive_and_closes_its_budget():
    manifest = json.loads((ROOT / "results" / "manifest.json").read_text())
    extra = manifest["model_exploration_20261017"]
    budget = extra["budget_closed"]

    assert extra["report"] == "experiments/model_exploration/REPORT.md"
    assert extra["protocol"] == "experiments/model_exploration/PROTOCOL.md"
    assert extra["results"] == "results/model-exploration-20261017/"
    assert extra["decision"] == "retain_existing_production_default"
    assert extra["production_default_changed"] is False
    assert extra["head_refinement"]["candidate_tag"] == "modernbert_head"
    assert extra["head_refinement"]["candidate_used"] is False
    assert budget["source"] == "results/model-exploration-20261017/budget_closed.json"
    assert budget["additional_gateway_spend_usd"] == pytest.approx(3.66106068)
    assert budget["uncached_list_estimate_usd"] == pytest.approx(4.24029855)
    assert budget["accounted_phase_liability_usd"] == pytest.approx(4.395242575)
    assert budget["additional_to_prior_key_usage"] is True
    assert manifest["claimed_from_run"] == M.CLAIM_RUN
