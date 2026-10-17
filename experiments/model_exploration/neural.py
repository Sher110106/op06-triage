"""Joint encoder candidate for OP-06.

Examples (run from the solution root):

  python experiments/model_exploration/neural.py train --model answerdotai/ModernBERT-base \
      --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 --tag modernbert_joint
  python experiments/model_exploration/neural.py predict --tag modernbert_joint \
      --splits selection calibration
  python experiments/model_exploration/neural.py train --model answerdotai/ModernBERT-base \
      --revision <pinned-hub-commit> --tag modernbert_bench --bench 8

Training uses only model_common.partition("fit") for gradient updates and the selection partition to pick the
best checkpoint. Calibration and confirmation labels are not read by this module's training path. `predict`
exports raw logits; the evaluator owns temperature fitting and final scoring.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from triage import text as T  # noqa: E402

try:
    from experiments.model_exploration import model_common as M  # noqa: E402
except ImportError as exc:  # gives a direct diagnostic when the coordinating helper is not in place yet
    raise ImportError("neural.py requires experiments/model_exploration/model_common.py") from exc


HEADS = ("intent", "flow", "action")
LOSS_WEIGHT = {"intent": 1.0, "flow": 0.5, "action": 1.0}
SPLITS = ("selection", "calibration", "confirmation", "dev", "noisy", "hostile")
SEED = 20261020
STATE_HIDDEN = 64
CHECKPOINT_FORMAT = 1


def _model_row(row: dict) -> dict:
    """Use the shared exploration input cleaning for text and dialogue-state features."""
    return M.model_input(row)


def _turn_segments(tokenizer, row: dict, max_turn_chars: int = 2000) -> list[list[int]]:
    segments = []
    for speaker, text in T.turns(_model_row(row)):
        if not text:
            continue
        # T.turns applies NFKC, control/format cleanup, lowercase and the 2,000-character cap.
        text = text[:max_turn_chars]
        ids = tokenizer(f"{speaker}: {text}\n", add_special_tokens=False)["input_ids"]
        if ids:
            segments.append(ids)
    return segments


def _special_prefix_suffix(tokenizer) -> tuple[list[int], list[int]]:
    """Derive the tokenizer's single-sequence wrapper using only its callable interface.

    Recent Transformers releases expose tokenizers that do not implement the older
    `build_inputs_with_special_tokens` method, while both forms support tokenizing a sample
    with and without special tokens.
    """
    full = tokenizer("hello", add_special_tokens=True)["input_ids"]
    inner = tokenizer("hello", add_special_tokens=False)["input_ids"]
    if not inner:
        raise ValueError("tokenizer produced no content tokens for the special-token probe")
    for start in range(len(full) - len(inner) + 1):
        if full[start:start + len(inner)] == inner:
            return list(full[:start]), list(full[start + len(inner):])
    raise ValueError("could not locate plain probe tokens inside the special-token sequence")


def encode_row(tokenizer, row: dict, max_length: int,
               special_tokens: tuple[list[int], list[int]] | None = None) -> list[int]:
    """Keep recent complete turns within the model budget; truncate only a single overlong newest turn."""
    prefix, suffix = special_tokens or _special_prefix_suffix(tokenizer)
    budget = max_length - len(prefix) - len(suffix)
    if budget < 1:
        raise ValueError(f"max_length {max_length} leaves no room for text tokens")
    kept: list[list[int]] = []
    used = 0
    for segment in reversed(_turn_segments(tokenizer, row)):
        if used + len(segment) <= budget:
            kept.append(segment)
            used += len(segment)
            continue
        if not kept:
            kept.append(segment[:budget])
        break
    content = [token_id for segment in reversed(kept) for token_id in segment]
    if len(content) > budget:
        raise AssertionError("content budgeting exceeded the tokenizer maximum length")
    return prefix + content + suffix


def encode_rows(tokenizer, rows: list[dict], max_length: int) -> list[list[int]]:
    special_tokens = _special_prefix_suffix(tokenizer)
    return [encode_row(tokenizer, row, max_length, special_tokens) for row in rows]


def _state_rows(rows: list[dict]) -> np.ndarray:
    return np.vstack([T.state_dense(_model_row(row)) for row in rows]).astype(np.float32)


def _labels_by_row(rows: list[dict], label_map: dict[str, list[str]]) -> dict[str, np.ndarray]:
    result = {}
    for head in HEADS:
        index = {label: i for i, label in enumerate(label_map[head])}
        try:
            result[head] = np.asarray([index[row[head]] for row in rows], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"{head} label {exc.args[0]!r} is absent from fit label ordering") from exc
    return result


def _batches(sequences: list[list[int]], indices, batch_size: int, shuffle: bool, rng: random.Random):
    indices = list(map(int, indices))
    if shuffle:
        rng.shuffle(indices)
        chunks = [indices[i:i + batch_size * 40] for i in range(0, len(indices), batch_size * 40)]
        batches = []
        for chunk in chunks:
            chunk.sort(key=lambda j: len(sequences[j]))
            batches.extend(chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size))
        rng.shuffle(batches)
    else:
        indices.sort(key=lambda j: len(sequences[j]))
        batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    return batches


def _tensor_batch(torch, sequences, batch_ids, pad_id, device):
    length = max(len(sequences[i]) for i in batch_ids)
    ids = torch.full((len(batch_ids), length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch_ids), length), dtype=torch.long)
    for k, row_idx in enumerate(batch_ids):
        seq = sequences[row_idx]
        ids[k, :len(seq)] = torch.as_tensor(seq, dtype=torch.long)
        mask[k, :len(seq)] = 1
    return ids.to(device), mask.to(device)


def _state_norm(state: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((state - mean) / std).astype(np.float32)


def _scatter_batch_logits(logits: dict[str, np.ndarray], batch_ids: list[int], batch_output: dict[str, Any],
                          row_count: int) -> None:
    """Place a length-bucketed batch back at its source-row offsets."""
    for head in HEADS:
        values = batch_output[head]
        if head not in logits:
            logits[head] = np.empty((row_count, values.shape[1]), dtype=values.dtype)
        logits[head][batch_ids] = values


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dirs(tag: str) -> tuple[Path, Path]:
    if not tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in tag):
        raise ValueError("tag must contain only letters, digits, hyphens and underscores")
    output_dir = Path(M.OUT) / tag
    checkpoint_dir = Path(M.ROOT) / "models" / "model-exploration" / tag
    return output_dir, checkpoint_dir


def _class_counts(label_map: dict[str, list[str]]) -> dict[str, int]:
    missing = [head for head in HEADS if not label_map.get(head)]
    if missing:
        raise ValueError(f"model_common.labels() has no classes for {missing}")
    return {head: len(label_map[head]) for head in HEADS}


def _make_model(name, revision, label_map, state_dim, *, config=None, load_pretrained: bool):
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel

    if config is None:
        config = AutoConfig.from_pretrained(name, revision=revision)
    if "modernbert" in name.lower():
        config.reference_compile = False
    attention = ("sdpa" if "modernbert" in name.lower()
                 and hasattr(torch.nn.functional, "scaled_dot_product_attention") else None)
    attention_kwargs = {"attn_implementation": attention} if attention else {}
    encoder = (AutoModel.from_pretrained(name, revision=revision, config=config, **attention_kwargs)
               if load_pretrained else AutoModel.from_config(config, **attention_kwargs))
    hidden = int(config.hidden_size)

    class JointModel(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.encoder = enc
            self.dropout = nn.Dropout(0.1)
            self.intent = nn.Linear(hidden, len(label_map["intent"]))
            self.flow = nn.Linear(hidden, len(label_map["flow"]))
            self.state_mlp = nn.Sequential(nn.Linear(state_dim, STATE_HIDDEN), nn.GELU(), nn.Dropout(0.1))
            self.action_hidden = nn.Sequential(
                nn.Linear(hidden + len(label_map["intent"]) + STATE_HIDDEN, hidden), nn.GELU(), nn.Dropout(0.1)
            )
            self.action = nn.Linear(hidden, len(label_map["action"]))

        def forward(self, input_ids, attention_mask, state_features):
            hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            weights = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            pooled = (hidden_states * weights).sum(1) / weights.sum(1).clamp(min=1.0)
            pooled = self.dropout(pooled)
            intent_logits = self.intent(pooled)
            flow_logits = self.flow(pooled)
            predicted_intent = torch.softmax(intent_logits, dim=-1)
            state_embedding = self.state_mlp(state_features)
            action_input = torch.cat((pooled, predicted_intent, state_embedding), dim=-1)
            action_logits = self.action(self.action_hidden(action_input))
            return {"intent": intent_logits, "flow": flow_logits, "action": action_logits}

    return JointModel(encoder), config


def _run_logits(model, sequences, state, row_count, batch_size, pad_id, device):
    import torch

    model.eval()
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
    logits: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for batch_ids in _batches(sequences, range(row_count), batch_size, False, random.Random(SEED)):
            ids, mask = _tensor_batch(torch, sequences, batch_ids, pad_id, device)
            out = model(ids, mask, state_t[batch_ids])
            batch_logits = {head: out[head].float().cpu().numpy() for head in HEADS}
            _scatter_batch_logits(logits, batch_ids, batch_logits, row_count)
    return logits


def _evaluate(model, sequences, state, rows, label_map, batch_size, pad_id, device):
    y = _labels_by_row(rows, label_map)
    arrays = _run_logits(model, sequences, state, len(rows), batch_size, pad_id, device)
    accuracy = {h: float((arrays[h].argmax(1) == y[h]).mean()) for h in HEADS}
    return accuracy, arrays


def _metadata(args, label_map, state_mean, state_std, config):
    import platform
    import torch
    import transformers
    return {
        "format": CHECKPOINT_FORMAT,
        "tag": args.tag,
        "model": args.model,
        "revision": args.revision,
        "seed": args.seed,
        "runtime": {"platform": platform.platform(), "torch": str(torch.__version__),
                    "transformers": transformers.__version__, "numpy": np.__version__,
                    "device": args.device, "threads": args.threads},
        "max_length": args.max_length,
        "state_dim": int(len(state_mean)),
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "labels": label_map,
        "architecture": {
            "pooling": "attention-mask mean",
            "heads": {"intent": len(label_map["intent"]), "flow": len(label_map["flow"]),
                      "action": len(label_map["action"])},
            "action_inputs": ["pooled_encoder", "predicted_intent_softmax", "normalized_state_mlp"],
            "state_hidden": STATE_HIDDEN,
            "dropout": 0.1,
            "attention": "sdpa" if "modernbert" in args.model.lower() else "auto",
            "compile": False,
        },
        "encoder_config": config.to_dict(),
    }


def _save_best(model, tokenizer, config, checkpoint_dir: Path, metadata: dict, score: float) -> None:
    import torch

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = checkpoint_dir / "tokenizer"
    config_dir = checkpoint_dir / "encoder_config"
    tokenizer.save_pretrained(tokenizer_dir)
    config.save_pretrained(config_dir)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "best_score": score, "metadata": metadata}, checkpoint_dir / "best.pt")
    (checkpoint_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def train(args) -> None:
    import torch
    from transformers import AutoTokenizer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.set_num_threads(args.threads)
    _seed_everything(args.seed)
    output_dir, checkpoint_dir = _dirs(args.tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    fit_rows = list(M.partition("fit"))
    selection_rows = list(M.partition("selection"))
    if not fit_rows or not selection_rows:
        raise ValueError("fit and selection partitions must both contain rows")
    label_map = M.labels()
    class_counts = _class_counts(label_map)
    fit_y = _labels_by_row(fit_rows, label_map)
    fit_state = _state_rows(fit_rows)
    state_mean = fit_state.mean(axis=0)
    state_std = fit_state.std(axis=0)
    state_std[state_std < 1e-6] = 1.0
    fit_state = _state_norm(fit_state, state_mean, state_std)
    selection_state = _state_norm(_state_rows(selection_rows), state_mean, state_std)

    print(json.dumps({"event": "load_model", "model": args.model, "revision": args.revision,
                      "device": args.device, "fit_rows": len(fit_rows), "selection_rows": len(selection_rows)},
                     separators=(",", ":")), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    fit_sequences = encode_rows(tokenizer, fit_rows, args.max_length)
    selection_sequences = encode_rows(tokenizer, selection_rows, args.max_length)
    lengths = np.asarray([len(s) for s in fit_sequences])
    print(json.dumps({"event": "fit_token_lengths", "mean": float(lengths.mean()),
                      "p95": float(np.percentile(lengths, 95)), "max": int(lengths.max()),
                      "at_cap_share": float((lengths >= args.max_length).mean())}), flush=True)
    model, config = _make_model(args.model, args.revision, label_map, fit_state.shape[1], load_pretrained=True)
    model.to(args.device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    batches_per_epoch = math.ceil(len(fit_rows) / args.batch_size)
    total_steps = max(1, int(batches_per_epoch * args.epochs))
    warmup_steps = max(1, int(0.06 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: ((step + 1) / warmup_steps if step < warmup_steps else
                      max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))),
    )
    fit_y_t = {h: torch.as_tensor(fit_y[h], dtype=torch.long, device=args.device) for h in HEADS}
    fit_state_t = torch.as_tensor(fit_state, dtype=torch.float32, device=args.device)
    eval_every = max(1, batches_per_epoch // args.evals_per_epoch)
    rng = random.Random(args.seed)
    start = time.monotonic()
    step = 0
    epoch = 0
    best_score = -1.0
    has_checkpoint = False
    evaluations = []
    bench_steps = args.bench
    stop = False

    while not stop:
        model.train()
        train_batches = _batches(fit_sequences, range(len(fit_rows)), args.batch_size, True, rng)
        for batch_ids in train_batches:
            ids, mask = _tensor_batch(torch, fit_sequences, batch_ids, pad_id, args.device)
            output = model(ids, mask, fit_state_t[batch_ids])
            loss = sum(LOSS_WEIGHT[h] * torch.nn.functional.cross_entropy(output[h], fit_y_t[h][batch_ids])
                       for h in HEADS)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            elapsed = time.monotonic() - start
            if step % 100 == 0:
                print(json.dumps({"event": "train_progress", "step": step,
                                  "elapsed_min": round(elapsed / 60, 2),
                                  "loss": round(float(loss.detach()), 5)}), flush=True)
            if bench_steps and step >= bench_steps:
                print(json.dumps({"event": "bench_done", "steps": step, "elapsed_s": round(elapsed, 2),
                                  "seconds_per_step": round(elapsed / step, 3),
                                  "examples_per_second": round(step * args.batch_size / elapsed, 2),
                                  "checkpoint_saved": False}, separators=(",", ":")), flush=True)
                return

            time_up = elapsed >= args.time_budget_min * 60
            if step % eval_every == 0 or step >= total_steps or time_up:
                accuracy, _ = _evaluate(model, selection_sequences, selection_state, selection_rows, label_map,
                                        args.batch_size, pad_id, args.device)
                score = accuracy["action"] + 0.25 * accuracy["intent"]
                event = {"step": step, "epoch": round(step / batches_per_epoch, 3), "elapsed_min": round(elapsed / 60, 2),
                         "selection_accuracy": accuracy, "selection_score": round(score, 6),
                         "lr": optimizer.param_groups[0]["lr"]}
                evaluations.append(event)
                print(json.dumps({"event": "selection_eval", **event}, separators=(",", ":")), flush=True)
                if score > best_score:
                    best_score = score
                    has_checkpoint = True
                    metadata = _metadata(args, label_map, state_mean, state_std, config)
                    _save_best(model, tokenizer, config, checkpoint_dir, metadata, best_score)
                    print(json.dumps({"event": "checkpoint_updated", "step": step,
                                      "selection_score": round(best_score, 6),
                                      "path": str(checkpoint_dir / "best.pt")}, separators=(",", ":")), flush=True)
                model.train()
                if time_up or step >= total_steps:
                    stop = True
                    break
        epoch += 1
        if epoch >= math.ceil(args.epochs):
            stop = True

    if not has_checkpoint:
        raise RuntimeError("training ended before a selection checkpoint could be written")
    # The best checkpoint was written at selection time. Persist only run metadata/logs alongside it.
    log = {
        "tag": args.tag,
        "model": args.model,
        "revision": args.revision,
        "device": args.device,
        "seed": args.seed,
        "threads": args.threads,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "epochs_requested": args.epochs,
        "steps": step,
        "elapsed_min": round((time.monotonic() - start) / 60, 2),
        "fit_rows": len(fit_rows),
        "selection_rows": len(selection_rows),
        "classes": class_counts,
        "best_selection_score": best_score,
        "selection_evaluations": evaluations,
        "checkpoint": str(checkpoint_dir / "best.pt"),
        "calibration_and_confirmation_used_for_training_or_selection": False,
    }
    (output_dir / "train.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")


def _load_checkpoint(tag: str, device: str):
    import torch
    from transformers import AutoConfig, AutoTokenizer

    output_dir, checkpoint_dir = _dirs(tag)
    checkpoint_path = checkpoint_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint missing: {checkpoint_path}; run the train command first")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    if metadata.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format: {metadata.get('format')!r}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer", local_files_only=True)
    config = AutoConfig.from_pretrained(checkpoint_dir / "encoder_config", local_files_only=True)
    model, _ = _make_model(metadata["model"], metadata["revision"], metadata["labels"],
                           metadata["state_dim"], config=config, load_pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, tokenizer, metadata, output_dir, device


def predict(args) -> None:
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model, tokenizer, metadata, output_dir, device = _load_checkpoint(args.tag, args.device)
    state_mean = np.asarray(metadata["state_mean"], dtype=np.float32)
    state_std = np.asarray(metadata["state_std"], dtype=np.float32)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    for split in args.splits:
        rows = list(M.partition(split))
        sequences = encode_rows(tokenizer, rows, metadata["max_length"])
        state = _state_norm(_state_rows(rows), state_mean, state_std)
        logits = _run_logits(model, sequences, state, len(rows), args.batch_size, pad_id, device)
        out_path = output_dir / f"logits_{split}.npz"
        np.savez_compressed(out_path, **logits)
        row_ids = [str(r["id"]) for r in rows]
        (output_dir / f"ids_{split}.json").write_text(json.dumps(row_ids) + "\n", encoding="utf-8")
        print(json.dumps({"event": "predictions_saved", "tag": args.tag, "split": split,
                          "rows": len(rows), "logits": str(out_path),
                          "ids": str(output_dir / f"ids_{split}.json")}, separators=(",", ":")), flush=True)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train", help="train on fit and pick one checkpoint on selection")
    tr.add_argument("--model", default="answerdotai/ModernBERT-base")
    tr.add_argument("--revision", required=True, help="pinned Hugging Face commit hash or immutable revision")
    tr.add_argument("--tag", required=True)
    tr.add_argument("--device", default="cpu")
    tr.add_argument("--threads", type=int, default=24)
    tr.add_argument("--epochs", type=float, default=3.0)
    tr.add_argument("--lr", type=float, default=3e-5)
    tr.add_argument("--bs", dest="batch_size", type=int, default=16)
    tr.add_argument("--maxlen", dest="max_length", type=int, default=512)
    tr.add_argument("--time-budget-min", type=float, default=45.0)
    tr.add_argument("--evals-per-epoch", type=int, default=2)
    tr.add_argument("--seed", type=int, default=SEED)
    tr.add_argument("--bench", type=int, choices=(8, 12), default=0,
                    help="run only 8 or 12 optimizer steps and do not save a checkpoint")
    tr.set_defaults(func=train)

    pr = sub.add_parser("predict", help="write logits and matching row ids for selected partitions")
    pr.add_argument("--tag", required=True)
    pr.add_argument("--splits", nargs="+", choices=SPLITS, default=["selection", "calibration"])
    pr.add_argument("--device", default="cpu")
    pr.add_argument("--bs", dest="batch_size", type=int, default=32)
    pr.set_defaults(func=predict)
    return ap


def main() -> None:
    args = parser().parse_args()
    if args.command == "train":
        if args.threads < 1 or args.batch_size < 1 or args.epochs <= 0 or args.lr <= 0:
            raise SystemExit("threads, batch size, epochs and learning rate must be positive")
        if args.max_length < 16 or args.time_budget_min <= 0 or args.evals_per_epoch < 1:
            raise SystemExit("maxlen must be >=16; time budget and evals-per-epoch must be positive")
    elif args.batch_size < 1:
        raise SystemExit("batch size must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
