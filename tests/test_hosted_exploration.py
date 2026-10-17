"""Offline regressions for the bounded hosted-model runner."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from experiments.model_exploration import hosted as H


MODEL = "provider/test-snapshot"
INTENT = H.P.INTENTS[0]
ACTION = H.P.ACTIONS[0]
ALT_ACTION = H.P.ACTIONS[1]


def _prices(*, endpoint_prompt: float = 2e-9, endpoint_completion: float = 3e-9) -> dict:
    return {
        "threshold_tiers": [{"min_prompt_tokens": 0, "prompt": 1e-9, "completion": 1e-9}],
        "time_tiers": [],
        "max_prompt_rate": 1e-9,
        "max_completion_rate": 1e-9,
        "endpoint_prompt_max": endpoint_prompt,
        "endpoint_completion_max": endpoint_completion,
        "reservation_prompt_rate": endpoint_prompt,
        "reservation_completion_rate": endpoint_completion,
    }


def _valid_gateway_response(action: str = ALT_ACTION, *, usage: dict | None = None) -> dict:
    content = json.dumps({"intent": INTENT, "action": action, "needs_human": False, "confidence": 0.8})
    return {"id": "gateway-1", "model": MODEL,
            "choices": [{"message": {"content": content}, "provider": "test"}],
            "usage": usage if usage is not None else {
                "prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16, "cost": 0.0}}


def _setup_run(monkeypatch, tmp_path, *, response: dict | None = None, run_id: str = "test-run",
               max_spend: float = 1.5):
    root = tmp_path / "repo"
    root.mkdir()
    out = root / "results"
    monkeypatch.setattr(H, "ROOT", root)
    monkeypatch.setattr(H, "OUT", out)
    monkeypatch.setattr(H, "_prices", lambda model_id: _prices())
    monkeypatch.setattr(H, "_meter", lambda key: {"limit": 10.0, "limit_remaining": 10.0, "usage": 0.0})
    monkeypatch.setattr(H, "_phase_liabilities", lambda: 0.0)
    H.M.write_json(out / "budget_before.json", {"limit": 10.0, "limit_remaining": 10.0, "usage": 0.0})

    cohort_path = tmp_path / "cohort.json"
    cohort = {"format": 1, "split": "selection", "static_prompt": "static prompt",
              "static_prompt_sha256": H._sha("static prompt"), "policy_version": "2026-10-17.2",
              "output_schema": {}, "items": [{
                  "id": "row-1", "reference": {"id": "row-1", "intent": INTENT,
                      "action": ACTION, "confidence": 0.6, "needs_human": False},
                  "dynamic_prompt": "customer says café", "policy_version": "2026-10-17.2",
                  "masked_actions": [], "retrieval_example_ids": [],
              }]}
    H.M.write_json(cohort_path, cohort)
    fallback = [{"id": "row-1", "intent": INTENT, "action": ACTION,
                 "confidence": 0.6, "needs_human": False}]
    H._append_jsonl(out / "fallback_selection.jsonl", fallback[0])
    key_path = tmp_path / "openrouter.key"
    key_path.write_text("offline-test-key", encoding="utf-8")
    args = SimpleNamespace(run_id=run_id, timeout=10, max_spend_usd=max_spend,
                            key_file=str(key_path), model=MODEL, cohort=str(cohort_path), limit=None)

    posts = []

    class FakeResponse:
        status = 200

        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, limit=-1):
            return json.dumps(self.payload).encode("utf-8")[:limit]

    def fake_urlopen(request, timeout):
        posts.append(request)
        assert request.full_url.endswith("/chat/completions")
        return FakeResponse(response if response is not None else _valid_gateway_response())

    monkeypatch.setattr(H.urllib.request, "urlopen", fake_urlopen)
    return out, cohort_path, args, posts


@pytest.mark.parametrize("output_tokens", [256, 1024])
def test_run_reserves_full_utf8_request_at_premium_endpoint_prices(monkeypatch, tmp_path, output_tokens):
    out, _, args, posts = _setup_run(monkeypatch, tmp_path)
    monkeypatch.setattr(H, "_prices", lambda model_id: _prices(endpoint_prompt=4e-9, endpoint_completion=5e-9))
    monkeypatch.setattr(H, "request_profile", lambda model_id: {
        "max_tokens": output_tokens, "reasoning": {"effort": "none"}, "temperature": 0})

    H._run(args)

    started = H._jsonl(out / "hosted" / args.run_id / "started_calls.jsonl")[0]
    request_json = json.dumps(started["request"], ensure_ascii=False, separators=(",", ":"))
    expected = ((len(request_json.encode("utf-8")) + H.OVERHEAD_BYTES) * 4e-9
                + output_tokens * 5e-9)
    assert len(posts) == 1
    assert started["reserved_usd"] == pytest.approx(expected)
    assert started["reserved_usd"] > ((len(request_json.encode("utf-8")) + H.OVERHEAD_BYTES) * 1e-9
                                       + output_tokens * 1e-9)


@pytest.mark.parametrize("gateway_response", [
    {"choices": [None]},
    {"choices": [{"message": "not-an-object"}]},
    {"choices": [{"message": {"content": json.dumps({"intent": INTENT, "action": ALT_ACTION,
       "needs_human": False, "confidence": 0.8})}}],
     "usage": {"prompt_tokens": -1, "completion_tokens": 1, "total_tokens": 0, "cost": -0.1}},
], ids=["null-choice", "non-object-message", "invalid-usage"])
def test_malformed_success_is_finalized_as_fallback_and_not_retried(
        monkeypatch, tmp_path, gateway_response):
    out, _, args, posts = _setup_run(monkeypatch, tmp_path, response=gateway_response)
    H._run(args)
    H._run(args)

    run_dir = out / "hosted" / args.run_id
    raw = H._jsonl(run_dir / "raw_calls.jsonl")
    prediction = H._jsonl(run_dir / "predictions_selection.jsonl")
    assert len(posts) == len(raw) == 1
    assert raw[0]["accepted_action"] is False
    assert prediction[0]["action"] == ACTION
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["new_attempts"] == 0


def test_started_ledger_prevents_reposting_after_interruption_before_final_record(monkeypatch, tmp_path):
    out, _, args, posts = _setup_run(monkeypatch, tmp_path)
    append = H._append_jsonl

    def interrupt_before_final(path, value):
        if path.name == "raw_calls.jsonl":
            raise RuntimeError("simulated process interruption after the POST")
        append(path, value)

    monkeypatch.setattr(H, "_append_jsonl", interrupt_before_final)
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        H._run(args)
    assert len(posts) == 1
    assert len(H._jsonl(out / "hosted" / args.run_id / "started_calls.jsonl")) == 1

    monkeypatch.setattr(H, "_append_jsonl", append)
    H._run(args)

    assert len(posts) == 1
    assert not (out / "hosted" / args.run_id / "raw_calls.jsonl").exists()
    assert H._jsonl(out / "hosted" / args.run_id / "predictions_selection.jsonl")[0]["action"] == ACTION


def test_per_run_budget_stop_writes_fallback_summary_without_raw_calls_file(monkeypatch, tmp_path):
    out, _, args, posts = _setup_run(monkeypatch, tmp_path, max_spend=1e-12)

    H._run(args)

    run_dir = out / "hosted" / args.run_id
    assert posts == []
    assert not (run_dir / "raw_calls.jsonl").exists()
    assert H._jsonl(run_dir / "predictions_selection.jsonl")[0]["action"] == ACTION
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["new_attempts"] == 0
    assert summary["stop_reason"] == "per_run_reservation_cap"


def test_final_predictions_use_frozen_fallback_after_reference_file_changes(monkeypatch, tmp_path):
    out, _, args, _ = _setup_run(monkeypatch, tmp_path)
    # A mutable reference artifact changes after cohort preparation; the frozen fallback is authoritative.
    H._append_jsonl(out / "reference_selection.jsonl", {
        "id": "row-1", "intent": H.P.INTENTS[-1], "action": ALT_ACTION,
        "confidence": 0.99, "needs_human": True})

    H._run(args)

    prediction = H._jsonl(out / "hosted" / args.run_id / "predictions_selection.jsonl")[0]
    assert prediction["intent"] == INTENT
    assert prediction["action"] == ALT_ACTION
    assert prediction["needs_human"] is True


def test_phase_liability_releases_completed_reservation_but_keeps_unresolved_one(monkeypatch, tmp_path):
    out = tmp_path / "results"
    monkeypatch.setattr(H, "OUT", out)
    prices = {"threshold_tiers": [{"min_prompt_tokens": 0, "prompt": 0.01, "completion": 0.02}],
              "time_tiers": []}
    monkeypatch.setattr(H, "_prices", lambda model_id: prices)
    complete_dir = out / "hosted" / "completed"
    unknown_dir = out / "hosted" / "interrupted"
    H._append_jsonl(complete_dir / "started_calls.jsonl", {"id": "done", "reserved_usd": 3.0})
    H._append_jsonl(complete_dir / "raw_calls.jsonl", {
        "id": "done", "model_requested": MODEL, "status": "ok", "reserved_usd": 3.0,
        "request_started_utc": "2026-10-18T00:00:00Z",
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110, "cost": 0.8}})
    H._append_jsonl(unknown_dir / "started_calls.jsonl", {"id": "unknown", "reserved_usd": 2.0})

    # The completed row is charged at max($0.80 gateway, $1.20 list estimate),
    # while the started-only row keeps its full $2 reservation.
    assert H._phase_liabilities() == pytest.approx(3.2)
