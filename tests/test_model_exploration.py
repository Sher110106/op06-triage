"""Regression checks for observed exploration defects; no transformer dependency or download."""
from __future__ import annotations

import numpy as np

from experiments.model_exploration import neural as N
from experiments.model_exploration.evaluate import accuracy_rule_pass, exact_paired


class TokenizerWithoutLegacyBuilder:
    """Transformers 5 tokenizers need not expose build_inputs_with_special_tokens."""
    def __call__(self, text, add_special_tokens=True):
        ids = [ord(c) + 10 for c in text]
        return {"input_ids": [1, *ids, 2] if add_special_tokens else ids}


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
