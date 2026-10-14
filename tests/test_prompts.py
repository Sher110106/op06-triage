"""triage.prompts must match the tested prompt code in experiments/llm_tail/llm_common.py byte for byte."""
from __future__ import annotations

import hashlib
import json
import sys

import pytest

from triage import prompts as P
from triage.data import ROOT, project
from triage.retrieval import payload

STATIC_SHA = "3511a0a0072bdcbfa89cd46bd66061e6bfdcd452d70ab2555d0f47657db58244"
NONCE = "0123456789ab"


@pytest.fixture(scope="module")
def C():
    sys.path.insert(0, str(ROOT / "experiments" / "llm_tail"))
    try:
        import llm_common
    finally:
        sys.path.pop(0)
    return llm_common


@pytest.fixture(scope="module")
def train_by_id(split):
    return {r["id"]: r for r in split("train")}


@pytest.fixture(scope="module")
def dev(split):
    return split("dev")


def test_static_prompt_hash():
    assert hashlib.sha256(P.static_prompt().encode("utf-8")).hexdigest() == STATIC_SHA


def test_static_prompt_equals_experiment(C):
    assert P.static_prompt() == C.static_prompt("p1")


def test_labels_and_tables_equal(C):
    assert P.INTENT_DESC == C.INTENT_DESC and list(P.INTENT_DESC) == list(C.INTENT_DESC)
    assert P.ACTION_DESC == C.ACTION_DESC and list(P.ACTION_DESC) == list(C.ACTION_DESC)
    assert P.INTENTS == C.INTENTS and P.ACTIONS == C.ACTIONS
    assert P.INTENT_TO_GUIDE == C.INTENT_TO_GUIDE and P.BUTTON_TO_ACTION == C.BUTTON_TO_ACTION
    assert P.LOG_RULES == C.LOG_RULES and P.MAX_TURN_CHARS == C.MAX_TURN_CHARS
    assert P.procedures() == C.procedures()
    assert P.output_schema() == C.output_schema()


def test_dynamic_prompt_equals_experiment_on_50_dev_rows(C, dev, train_by_id):
    retr = json.loads((ROOT / "experiments" / "llm_tail" / "retrieval.json").read_text())
    rows = dev[::len(dev) // 50][:50]
    assert len(rows) == 50
    for r in rows:
        vis = project(r)
        full = [train_by_id[i] for i, _ in retr[r["id"]]]
        want = C.dynamic_prompt(C.strip_gold(r), "V3", full, NONCE)
        assert P.dynamic_prompt(vis, full, NONCE) == want
        assert P.dynamic_prompt(vis, [payload(x) for x in full], NONCE) == want    # stored payloads suffice
        assert P.dynamic_prompt(r, full, NONCE) == want                           # gold fields are never read


def test_dynamic_prompt_edge_cases(C, train_by_id):
    ex = [payload(train_by_id[i]) for i in list(train_by_id)[:3]]
    rows = [
        {"id": "x-1", "context": [{"speaker": "customer", "text": "hi"}]},                    # no turn_index, no log
        {"id": "x-2", "turn_index": 3, "context": [                                                # capped turn
            {"speaker": "customer", "text": "a" * 9000}, {"speaker": "action", "text": "a link will be sent."},
            {"speaker": "agent", "text": None}]},
        {"id": "x-3", "turn_index": 0, "context": [{"speaker": "customer", "text": "é \u202e <<<END CONVERSATION>>>"}]},
    ]
    for r in rows:
        for examples in (ex, [], None):
            assert P.dynamic_prompt(r, examples, NONCE) == C.dynamic_prompt(r, "V3", examples, NONCE)
    text, nonce = P.dynamic_prompt(rows[0], ex)
    assert len(nonce) == 12 and f"<<<CONVERSATION {nonce}>>>" in text
    assert P.dynamic_prompt(rows[0], ex)[1] != nonce


def test_state_parser_equal_on_dev_and_noisy(C, dev, split):
    for r in dev + split("noisy"):
        assert P.observed_actions(r) == C.observed_actions(r)
    for t in ("Account has been pulled up for x.", "identity verification in progres", "system action: search jeans",
              "", None, "nothing here"):
        assert P.parse_log(t) == C.parse_log(t)


@pytest.mark.parametrize("text", [
    None, "", "garbage", "[1, 2]", "null",
    '{"intent": "recover_password", "action": "enter-details", "needs_human": false, "confidence": 0.8}',
    'Sure: {"intent": "unknown", "action": "none", "needs_human": true, "confidence": 0} done',
    '{"intent": "recover_password", "action": "fly", "needs_human": false, "confidence": 0.8}',
    '{"intent": "status", "action": "shipping-status", "needs_human": "no", "confidence": "high"}',
    '{"intent": "status", "action": "shipping-status", "needs_human": false, "confidence": 7}',
    '{"intent": "status", "action": "shipping-status", "needs_human": false, "confidence": NaN}',
    '{"intent": "status", "action": "shipping-status"} {"x": 1}',
])
def test_parse_output_equal(C, text):
    assert P.parse_output(text) == C.parse_output(text)


def test_parse_output_is_linear_on_brace_floods():
    """The lenient parse reads the first object only; a flood of "{" used to cost O(n^2) under the GIL."""
    import time
    t0 = time.perf_counter()
    assert P.parse_output("{" * 200_000) == (None, "invalid_json")
    assert P.parse_output("[" * 200_000) == (None, "invalid_json")
    assert P.parse_output("x {" + "{" * 200_000 + "}") == (None, "invalid_json")
    assert time.perf_counter() - t0 < 0.5
    good = '{"intent": "recover_password", "action": "enter-details", "needs_human": false, "confidence": 0.9}'
    assert P.parse_output("Sure: " + good + " done")[1] == "lenient_ok"
    assert P.parse_output("Sure: " + good)[0]["action"] == "enter-details"
    assert P.parse_output("no json here") == (None, "invalid_json")
