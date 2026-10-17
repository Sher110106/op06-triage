"""Regression coverage for human review after an LLM changes the proposed action."""
from __future__ import annotations

import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from triage.classical import DEFAULT_BUNDLE
from triage.data import load_split, project
from triage.llm_action import LLMResult
from triage.pipeline import Triage
from triage.policy import DEFAULT_POLICY, PolicyStore, load
from triage.service import Service, trace_fields
from triage.text import ACTIONS30, observed_actions


class FixedLLM:
    """Minimal LLM step double used through the real Triage request path."""

    def __init__(self, action: str):
        self.cfg = SimpleNamespace(gate=1.0)
        self.result = LLMResult("ok", action=action)
        self.calls = 0

    def decide(self, row):
        self.calls += 1
        return self.result


@pytest.fixture(scope="module")
def scenario():
    if not DEFAULT_BUNDLE.exists():
        pytest.skip("models/classical_v0.joblib is not built")

    policy = load(DEFAULT_POLICY)
    with tempfile.TemporaryDirectory(prefix="triage-refinement-") as tmp:
        policy_path = Path(tmp) / "zero-thresholds.toml"
        text = DEFAULT_POLICY.read_text(encoding="utf-8")
        text, n = re.subn(r"(?m)^(high|low) = [0-9.]+$", r"\1 = 0.0", text)
        assert n == 2
        policy_path.write_text(text, encoding="utf-8")

        triage = Triage(bundle=DEFAULT_BUNDLE, policy_path=policy_path)
        try:
            rows = [project(row) for row in load_split("dev")]
            outputs = triage.handle_many(rows)
            for row, (response, trace) in zip(rows, outputs):
                flow = trace.get("cpu", {}).get("flow")
                permitted = policy.flow_permitted.get(flow, frozenset())
                if (trace["flags"] or trace["route_reasons"] or response["confidence"] >= 1.0
                        or response["action"] not in permitted):
                    continue
                alternatives = sorted((permitted & set(ACTIONS30)) - {response["action"]})
                if alternatives:
                    yield triage, row, response, trace, alternatives[0], policy_path
                    break
            else:
                pytest.fail("no dev row has a clean, permitted CPU action and a permitted alternative")
        finally:
            triage.close()


def test_llm_changed_action_always_routes_for_review(scenario):
    """A permitted LLM change still needs review, even when confidence thresholds are zero."""
    triage, row, cpu_response, _, alternative, _ = scenario
    fake = FixedLLM(alternative)
    triage.llm = fake

    response, trace = triage.handle(row)

    assert fake.calls == 1
    assert response["action"] == alternative
    assert response["action"] != cpu_response["action"]
    assert response["needs_human"] is True
    assert "llm_disagreement" in trace["route_reasons"]
    assert trace["llm"]["used"] is True
    assert trace["llm"]["agrees_with_cpu"] is False
    assert trace["llm"]["rejected_by_mask"] is False


def test_llm_agreement_does_not_force_review(scenario):
    """An agreeing, permitted LLM result remains automatic when the confidence cutoff permits it."""
    triage, row, cpu_response, _, _, _ = scenario
    fake = FixedLLM(cpu_response["action"])
    triage.llm = fake

    response, trace = triage.handle(row)

    assert fake.calls == 1
    assert response["action"] == cpu_response["action"]
    assert response["needs_human"] is False
    assert "llm_disagreement" not in trace["route_reasons"]
    assert trace["llm"]["used"] is True
    assert trace["llm"]["agrees_with_cpu"] is True
    assert trace["llm"]["rejected_by_mask"] is False


def test_masked_llm_action_is_traced_but_not_used(scenario):
    """A masked LLM suggestion is observable, while the CPU action remains in force."""
    triage, row, cpu_response, _, alternative, path = scenario
    required = next(a for a in ACTIONS30 if a not in set(observed_actions(row)))
    rule = (f'\n[[rules]]\nname = "mask-llm-choice"\naction = "{alternative}"\n'
            f'requires_any_observed = ["{required}"]\neffect = "mask"\n')
    original_text = path.read_text(encoding="utf-8")
    original_store = triage.store
    try:
        path.write_text(original_text + rule, encoding="utf-8")
        triage.store = PolicyStore(path, check_every_s=0.0)
        fake = FixedLLM(alternative)
        triage.llm = fake

        response, trace = triage.handle(row)

        assert fake.calls == 1
        assert response["action"] == cpu_response["action"]
        assert response["needs_human"] is False
        assert trace["llm"]["used"] is False
        assert trace["llm"]["agrees_with_cpu"] is False
        assert trace["llm"]["rejected_by_mask"] is True
        assert "llm_disagreement" not in trace["route_reasons"]
    finally:
        path.write_text(original_text, encoding="utf-8")
        triage.store = original_store


def test_service_counts_llm_decisions_once_even_on_replay():
    """Decision metrics distinguish accepted agreement/disagreement from mask rejection."""
    service = Service(triage=object(), loader=lambda: None, max_body=1024, idem_size=10)
    cases = [
        ("accepted_agreement", {"status": "ok", "used": True, "agrees_with_cpu": True,
                                "rejected_by_mask": False}),
        ("accepted_disagreement", {"status": "ok", "used": True, "agrees_with_cpu": False,
                                    "rejected_by_mask": False}),
        ("rejected_by_mask", {"status": "ok", "used": False, "agrees_with_cpu": False,
                              "rejected_by_mask": True}),
    ]
    for decision, llm in cases:
        info = trace_fields({"llm": llm})
        assert info["llm_decision"] == decision
        service.record(200, 1.0, {**info, "replay": False})
        service.record(200, 1.0, {**info, "replay": True})

    assert service.metrics()["llm"]["decisions"] == {
        "accepted_agreement": 1,
        "accepted_disagreement": 1,
        "rejected_by_mask": 1,
    }


def test_non_decision_llm_traces_do_not_increment_decision_counters():
    """Failures, abstentions and incomplete success traces are not accepted decisions."""
    service = Service(triage=object(), loader=lambda: None, max_body=1024, idem_size=10)
    traces = [
        {"status": "timeout"},
        {"status": "http_error"},
        {"status": "invalid"},
        {"status": "abstained"},
        {"status": "breaker_open"},
        {"status": "busy"},
        {"status": "ok", "used": True, "agrees_with_cpu": None, "rejected_by_mask": False},
        {"status": "ok", "used": 1, "agrees_with_cpu": True, "rejected_by_mask": False},
        {"status": "ok", "used": False, "agrees_with_cpu": False, "rejected_by_mask": False},
    ]

    for llm in traces:
        info = trace_fields({"llm": llm})
        assert info["llm_decision"] is None
        service.record(200, 1.0, {**info, "replay": False})

    assert service.metrics()["llm"]["decisions"] == {}
