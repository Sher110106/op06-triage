"""Offline end-to-end smoke for action-head refinement with a randomly initialized tiny BERT.

This verifies local checkpoint/tokenizer save and reload, frozen fit/selection boundaries,
feature-cache extraction, and the head-only refinement path. It never loads Hub weights or
reads the project's real partitions. Run it in an environment with torch, transformers,
tokenizers, numpy and scikit-learn installed:

    python -m experiments.model_exploration.smoke_head
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.model_exploration import model_common as M
from experiments.model_exploration import neural as N
from experiments.model_exploration import refine_head as R


def _tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    vocab.update({f"unused-{i}": i + 5 for i in range(27)})
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                   unk_token="[UNK]", cls_token="[CLS]",
                                   sep_token="[SEP]", mask_token="[MASK]")


def _rows(prefix: str, count: int, *, selection: bool = False) -> list[dict]:
    rows = []
    for i in range(count):
        rows.append({
            "id": f"{prefix}-{i}-1",
            "turn_index": 1,
            "context": [{"speaker": "customer", "text": f"please help with sample issue {prefix} {i}"}],
            "intent": "intent-a" if i % 2 == 0 else "intent-b",
            "flow": "flow-a" if i % 2 == 0 else "flow-b",
            "action": "action-a1" if selection or i > 0 else "action-a0",
        })
    return rows


def run_smoke() -> dict:
    import torch
    from transformers import BertConfig

    fit = _rows("fit", 8)
    selection = _rows("selection", 4, selection=True)
    labels = {"intent": ["intent-a", "intent-b"],
              "flow": ["flow-a", "flow-b"],
              "action": ["action-a0", "action-a1"]}
    partitions_seen: list[str] = []

    def partition(name: str) -> list[dict]:
        partitions_seen.append(name)
        if name == "fit":
            return fit
        if name == "selection":
            return selection
        raise AssertionError(f"refinement accessed forbidden partition {name!r}")

    with tempfile.TemporaryDirectory(prefix="op06-head-smoke-") as temp:
        repo = Path(temp) / "repo"
        output = repo / "results" / "model-exploration-20261017"
        model_dir = repo / "models" / "model-exploration"
        output.mkdir(parents=True)
        model_dir.mkdir(parents=True)

        tokenizer = _tokenizer()
        config = BertConfig(vocab_size=32, hidden_size=8, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=16,
                            max_position_embeddings=64, pad_token_id=0)
        model, config = N._make_model("unit-test/tiny-bert", "local", labels,
                                      len(N._state_rows(fit)[0]), config=config,
                                      load_pretrained=False)
        # Make the epoch-zero action prediction deterministic (class 0); training may then
        # learn the selection set's class 1 without asserting a brittle accuracy threshold.
        with torch.no_grad():
            model.action.weight.zero_()
            model.action.bias.zero_()

        state_fit = N._state_rows(fit)
        state_mean = state_fit.mean(axis=0)
        state_std = state_fit.std(axis=0)
        state_std[state_std < 1e-6] = 1.0
        base_args = type("Args", (), {"tag": "tiny-base", "model": "unit-test/tiny-bert",
            "revision": "local", "seed": 7, "device": "cpu", "threads": 1,
            "max_length": 32})()
        metadata = N._metadata(base_args, labels, state_mean, state_std, config)
        with (patch.object(M, "ROOT", repo), patch.object(M, "OUT", output),
              patch.object(M, "MODEL_DIR", model_dir), patch.object(M, "partition", partition),
              patch.object(M, "labels", lambda: labels)):
            # _dirs resolves paths from the patched M root; save only local random weights.
            base_checkpoint = model_dir / "tiny-base"
            N._save_best(model, tokenizer, config, base_checkpoint, metadata, 0.0)

            before, _, _, _, _ = N._load_checkpoint("tiny-base", "cpu")
            frozen_names = [name for name in before.state_dict()
                            if name.startswith(("encoder.", "intent.", "flow."))]
            frozen_before = {name: before.state_dict()[name].clone() for name in frozen_names}

            args = type("Args", (), {"base_tag": "tiny-base", "tag": "tiny-refined",
                "threads": 1, "head_threads": 1, "feature_bs": 4, "epochs": 2,
                "bs": 4, "lr": 0.01, "device": "cpu"})()
            R.refine(args)

            refined, tokenizer2, refined_meta, refined_output, _ = N._load_checkpoint("tiny-refined", "cpu")
            assert (refined_output / "head_refinement.json").is_file()
            assert (base_checkpoint / "best.pt").is_file()
            assert (model_dir / "tiny-refined" / "best.pt").is_file()
            for name, value in frozen_before.items():
                assert torch.equal(value, refined.state_dict()[name]), f"frozen tensor changed: {name}"

            raw_state = N._state_rows(selection)
            mean = np.asarray(refined_meta["state_mean"], dtype=np.float32)
            std = np.asarray(refined_meta["state_std"], dtype=np.float32)
            assert np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()
            normalized = N._state_norm(raw_state, mean, std)
            sequences = N.encode_rows(tokenizer2, selection, refined_meta["max_length"])
            logits = N._run_logits(refined, sequences, normalized, len(selection), 4,
                                   tokenizer2.pad_token_id, "cpu")
            for head in N.HEADS:
                assert logits[head].shape == (len(selection), len(labels[head]))
                assert np.isfinite(logits[head]).all(), f"non-finite {head} logits"
                probabilities = torch.softmax(torch.as_tensor(logits[head]), dim=-1).numpy()
                assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

            refinement = json.loads((refined_output / "head_refinement.json").read_text(encoding="utf-8"))
            assert refinement["cache"]["fit"] == "built"
            assert refinement["cache"]["selection"] == "built"
            assert refinement["cache"]["fit_max_forward_diff"] < 1e-4
            assert refinement["cache"]["selection_max_forward_diff"] < 1e-4
            assert set(partitions_seen) == {"fit", "selection"}
            assert len(partitions_seen) == 2
            return {"status": "passed", "epochs": args.epochs,
                    "fit_rows": len(fit), "selection_rows": len(selection),
                    "heads": {head: list(logits[head].shape) for head in N.HEADS},
                    "cache": refinement["cache"], "partition_calls": partitions_seen,
                    "checkpoint_reload": True, "network_calls": 0}


if __name__ == "__main__":
    print(json.dumps(run_smoke(), indent=2, sort_keys=True))
