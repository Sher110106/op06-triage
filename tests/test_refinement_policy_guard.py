"""Policy-permission support review tests for the real triage pipeline."""
from __future__ import annotations

import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from triage import gate
from triage.classical import DEFAULT_BUNDLE
from triage.data import load_split, project
from triage.llm_action import LLMResult
from triage.pipeline import Triage
from triage.policy import DEFAULT_POLICY, PolicyError, PolicyStore, load, loads
from triage.text import ACTIONS30

HIGH_RISK = {"offer-refund", "update-order", "update-account", "make-purchase", "promo-code", "make-password"}


def _base_policy_text() -> str:
    """Use the shipped procedure and disable confidence routing for clean before/after comparisons."""
    text = DEFAULT_POLICY.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^permission_review\s*=.*\n", "", text)
    head, sep, tail = text.partition("[thresholds]")
    assert sep
    tail, n = re.subn(r"(?m)^(high|low) = [0-9.]+$", r"\1 = 0.0", tail)
    assert n == 2
    return head + sep + tail


def _with_permission_review(text: str, settings: str) -> str:
    """Set the top-level permission-review table without duplicating a shipped value."""
    text = re.sub(r"(?m)^permission_review\s*=.*\n", "", text)
    at = text.index("\n[flows]")
    return text[:at] + f"\npermission_review = {settings}\n" + text[at:]


class FixedLLM:
    def __init__(self, action: str):
        self.cfg = SimpleNamespace(gate=1.0)
        self.result = LLMResult("ok", action=action)

    def decide(self, row):
        return self.result


@pytest.fixture(scope="module")
def examples():
    """Find real dev rows where high-risk permission support is uncertain and a low-risk control exists."""
    if not DEFAULT_BUNDLE.exists():
        pytest.skip("models/classical_v0.joblib is not built")

    with tempfile.TemporaryDirectory(prefix="triage-permission-guard-") as tmp:
        path = Path(tmp) / "policy.toml"
        base_text = _base_policy_text()
        path.write_text(base_text, encoding="utf-8")
        triage = Triage(bundle=DEFAULT_BUNDLE, policy_path=path)
        try:
            policy = load(path)
            rows = [project(row) for row in load_split("dev")]
            cleaned = []
            for row in rows:
                result = gate.clean(row)
                cleaned.append({"id": result.id, "turn_index": result.turn_index, "context": result.context})
            outputs = triage.handle_many(rows)
            _, _, flow_probs = triage.cpu.raw_probs(cleaned)

            high = low = None
            for i, (row, (response, trace)) in enumerate(zip(rows, outputs)):
                if trace["flags"] or trace["route_reasons"] or response["confidence"] >= 1.0:
                    continue
                action = response["action"]
                flow = trace["cpu"]["flow"]
                if action not in policy.flow_permitted[flow]:
                    continue
                def permission_mass(candidate: str) -> float:
                    return sum(float(flow_probs[i, j]) for j, f in enumerate(triage.cpu.fcls)
                               if candidate in policy.flow_permitted[str(f)])

                support = permission_mass(action)
                record = (row, response, trace, support, flow)
                if action in HIGH_RISK and support < 0.9 and high is None:
                    permitted_low = [candidate for candidate in sorted(
                        (policy.flow_permitted[flow] & set(ACTIONS30)) - HIGH_RISK)
                        if permission_mass(candidate) < 0.9]
                    if permitted_low:
                        high = (*record, permitted_low[0])
                elif action not in HIGH_RISK and support < 0.9 and low is None:
                    low = record
                if high is not None and low is not None:
                    break
            if high is None or low is None:
                pytest.fail("could not find suitable clean high-risk and low-risk dev examples")
            yield triage, path, base_text, high, low
        finally:
            triage.close()


def _install(triage: Triage, path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    triage.store = PolicyStore(path, check_every_s=0.0)


def test_high_risk_low_permission_support_routes_but_low_risk_is_unchanged(examples):
    triage, path, base_text, high, low = examples
    high_row, high_base, _, high_support, _ = high[:5]
    low_row, low_base, _, low_support, _ = low
    assert high_base["action"] in HIGH_RISK and high_support < 0.9

    original_store = triage.store
    guard_text = _with_permission_review(base_text, "{ high = 0.9 }")
    try:
        _install(triage, path, guard_text)
        high_guarded, high_trace = triage.handle(high_row)
        low_guarded, low_trace = triage.handle(low_row)

        assert high_guarded["intent"] == high_base["intent"]
        assert high_guarded["action"] == high_base["action"]
        assert high_guarded["needs_human"] is True
        assert "policy_uncertainty" in high_trace["route_reasons"]
        assert high_trace["route_reasons"] == ["policy_uncertainty"]
        assert high_guarded["confidence"] == high_base["confidence"]
        assert high_trace["cpu"]["final_action_permission_support"] == pytest.approx(high_support, abs=1e-6)
        assert "policy_uncertainty" not in low_trace["route_reasons"]
        assert low_guarded["action"] == low_base["action"]
        assert low_guarded["needs_human"] is low_base["needs_human"] is False
        assert low_support < 0.9
    finally:
        _install(triage, path, base_text)
        triage.store = original_store


def test_empty_permission_review_preserves_legacy_route(examples):
    triage, path, base_text, high, _ = examples
    row, baseline, _, _, _ = high[:5]
    original_store = triage.store
    try:
        _install(triage, path, _with_permission_review(base_text, "{}"))
        response, trace = triage.handle(row)
        assert response["action"] == baseline["action"]
        assert response["intent"] == baseline["intent"]
        assert response["needs_human"] is baseline["needs_human"] is False
        assert trace["cpu"]["final_action_permission_support"] == pytest.approx(high[3], abs=1e-6)
        assert "policy_uncertainty" not in trace["route_reasons"]
    finally:
        _install(triage, path, base_text)
        triage.store = original_store


def test_permission_support_is_checked_for_final_llm_action(examples):
    triage, path, base_text, high, _ = examples
    row, _, _, _, flow, low_action = high
    original_store, original_llm = triage.store, triage.llm
    try:
        _install(triage, path, _with_permission_review(base_text, "{ high = 0.9 }"))
        triage.llm = FixedLLM(low_action)
        response, trace = triage.handle(row)

        assert response["action"] == low_action
        assert "llm_disagreement" in trace["route_reasons"]
        assert "policy_uncertainty" not in trace["route_reasons"]
        assert trace["cpu"]["final_action_permission_support"] < 0.9
        assert flow in load(path).flow_permitted
    finally:
        triage.llm = original_llm
        _install(triage, path, base_text)
        triage.store = original_store


@pytest.mark.parametrize("setting", [
    "{ unknown = 0.9 }",
    "{ high = nan }",
    "{ high = 1.01 }",
])
def test_permission_review_loader_rejects_unknown_risk_and_invalid_threshold(setting):
    text = _with_permission_review(_base_policy_text(), setting)
    with pytest.raises(PolicyError):
        loads(text)


def test_permission_review_loader_exposes_tier_minimum():
    policy = loads(_with_permission_review(_base_policy_text(), "{ high = 0.9 }"))
    assert policy.minimum_permission_support("offer-refund") == pytest.approx(0.9)
    assert policy.minimum_permission_support("search-faq") == 0.0
