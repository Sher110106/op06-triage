"""Regression checks for observed exploration defects; no transformer dependency or download."""
from __future__ import annotations

import numpy as np

from experiments.model_exploration import neural as N
from experiments.model_exploration.evaluate import accuracy_rule_pass, exact_paired, score_rows
from harness.eval import Scored
from triage import prompts as P


class TokenizerWithoutLegacyBuilder:
    """Transformers 5 tokenizers need not expose build_inputs_with_special_tokens."""
    def __call__(self, text, add_special_tokens=True):
        ids = [ord(c) + 10 for c in text]
        return {"input_ids": [1, *ids, 2] if add_special_tokens else ids}


def test_hostile_null_targets_are_not_labelled_accuracy_rows():
    rows = [{"id": "hostile-null", "intent": None, "action": None, "flow": None}]
    assert score_rows(rows, {}, object()) == {"rows": 1}


def test_token_budget_and_gold_fields_are_isolated():
    row = {"id": "1-9", "turn_index": 9, "context": [
        {"speaker": "customer", "text": "old " * 200},
        {"speaker": "action", "text": "system action: verify identity"},
        {"speaker": "customer", "text": "I want a refund."}],
        "intent": "secret_target", "action": "future_action", "flow": "gold_flow"}
    changed = {**row, "intent": "different", "action": "different", "flow": "different"}
    tok = TokenizerWithoutLegacyBuilder()
    ids = N.encode_row(tok, row, 128)
    assert len(ids) <= 128 and ids[0] == 1 and ids[-1] == 2
    assert ids == N.encode_row(tok, changed, 128)
    assert np.array_equal(N._state_rows([row]), N._state_rows([changed]))
    decoded = "".join(chr(i - 10) for i in ids[1:-1])
    assert "i want a refund" in decoded and "old old" not in decoded


def test_length_bucket_outputs_restore_original_row_order():
    sequences = [[5] * 9, [7], [8] * 4]
    output = {}
    import random
    for batch in N._batches(sequences, range(3), 1, False, random.Random(0)):
        values = {head: np.asarray([[i, i + 10] for i in batch], dtype=np.float32) for head in N.HEADS}
        N._scatter_batch_logits(output, batch, values, 3)
    assert output["action"].tolist() == [[0, 10], [1, 11], [2, 12]]


def test_display_rounding_cannot_pass_the_one_point_rule():
    class Policy:
        risk = {}
        def _canon(self, a): return a
        def _permitted(self, a, f): return True
    rows = [{"id": f"{i}-1", "intent": "i", "action": "a", "flow": "f"} for i in range(3030)]
    baseline = {r["id"]: {"intent": "i", "action": "a" if i < 278 else "b", "needs_human": False}
                for i, r in enumerate(rows)}
    candidate = {r["id"]: {"intent": "i", "action": "a" if i < 308 else "b", "needs_human": False}
                 for i, r in enumerate(rows)}
    delta = exact_paired(rows, baseline, candidate, Policy(), draws=100)
    assert np.isclose(delta["action_accuracy"]["diff"], 30 / 3030)
    assert not accuracy_rule_pass(delta)


def test_noisy_only_metrics_leave_clean_accuracy_and_gap_undefined():
    rows = [
        {"id": "conversation-a-1-noisy", "_group": "noisy", "intent": P.INTENTS[0], "action": P.ACTIONS[0]},
        {"id": "conversation-b-1-noisy", "_group": "noisy", "intent": P.INTENTS[1], "action": P.ACTIONS[1]},
    ]
    preds = {
        row["id"]: {"id": row["id"], "intent": P.INTENTS[0], "action": P.ACTIONS[0],
                    "confidence": 0.8, "needs_human": False}
        for row in rows
    }

    metrics = Scored(rows, preds).metrics()

    assert metrics["action_acc_clean"] is None
    assert metrics["action_acc_noisy"] == 0.5
    assert metrics["noisy_gap"] is None

    clean = {"id": "conversation-clean-1", "_group": "clean",
             "intent": P.INTENTS[0], "action": P.ACTIONS[0]}
    mixed_preds = {**preds, clean["id"]: {"id": clean["id"], "intent": clean["intent"],
                   "action": clean["action"], "confidence": 0.8, "needs_human": False}}
    mixed = Scored([clean, *rows], mixed_preds).metrics()
    assert mixed["action_acc_clean"] == 1.0
    assert mixed["action_acc_noisy"] == 0.5
    assert mixed["noisy_gap"] == 0.5
