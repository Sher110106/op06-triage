"""Refine only the joint encoder's action/state head, keeping the backbone frozen.

  python experiments/model_exploration/refine_head.py --base-tag modernbert_joint \
      --tag modernbert_head --threads 32 --head-threads 4 --epochs 20 --bs 256 --lr 1e-3 --device cpu

Only the fit and selection partitions are read. Encoder representations are cached under ignored `models/` and
reused only when base-tag and row IDs match. This experiment does not read calibration or confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.model_exploration import model_common as M  # noqa: E402
from experiments.model_exploration import neural as N  # noqa: E402


SEED = 20261025
FEATURE_VERSION = 1


def _cache_paths(base_tag: str, split: str):
    cache = Path(M.MODEL_DIR) / f"{base_tag}-head-features"
    return (cache, cache / f"ids_{split}.json", cache / f"pooled_{split}.npy",
            cache / f"state_{split}.npy", cache / f"cache_meta_{split}.json")


def _producer_identity(base_tag: str) -> str:
    """Hash the exact model/config/tokenizer files that produced cached encoder features."""
    root = Path(M.MODEL_DIR) / base_tag
    paths = [root / "best.pt", root / "config.json"]
    paths += sorted((root / "encoder_config").rglob("*"))
    paths += sorted((root / "tokenizer").rglob("*"))
    paths = [p for p in paths if p.is_file()]
    if len(paths) < 4:
        raise FileNotFoundError(f"base checkpoint package incomplete: {root}")
    digest = hashlib.sha256(f"head-features-v{FEATURE_VERSION}\n{base_tag}\n".encode())
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _pool_and_check(model, sequences, state, rows, batch_size, pad_id, device):
    """Cache pre-dropout mean pool features and verify the frozen-head feature path against full forward."""
    import torch

    model.eval()
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
    pooled = None
    max_diff = 0.0
    with torch.no_grad():
        for batch_num, batch_ids in enumerate(
            N._batches(sequences, range(len(rows)), batch_size, False, random.Random(SEED))
        ):
            ids, mask = N._tensor_batch(torch, sequences, batch_ids, pad_id, device)
            hs = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).to(hs.dtype)
            p = (hs * m).sum(1) / m.sum(1).clamp(min=1.0)
            if pooled is None:
                pooled = np.empty((len(rows), p.shape[1]), dtype=np.float32)
            pooled[batch_ids] = p.float().cpu().numpy()

            if batch_num == 0:
                direct = model(ids, mask, state_t[batch_ids])
                head = _head_logits(model, p, state_t[batch_ids])
                max_diff = max(float((direct[k] - head[k]).abs().max().cpu()) for k in N.HEADS)
                if max_diff >= 1e-4:
                    raise RuntimeError(f"cached-feature head path differs from frozen model forward by {max_diff:.6g}")
    if pooled is None:
        raise ValueError("cannot cache features for an empty partition")
    return pooled, max_diff


def _head_logits(model, pooled, state):
    """Run just frozen intent/flow and candidate action head from pre-dropout pooled features."""
    import torch

    dropped = model.dropout(pooled)
    intent = model.intent(dropped)
    flow = model.flow(dropped)
    intent_prob = torch.softmax(intent, dim=-1)  # internal action feature is fixed T=1
    state_embedding = model.state_mlp(state)
    action_input = torch.cat((dropped, intent_prob, state_embedding), dim=-1)
    action = model.action(model.action_hidden(action_input))
    return {"intent": intent, "flow": flow, "action": action}


def _labels(rows, labels):
    lookup = {name: {label: i for i, label in enumerate(labels[name])} for name in N.HEADS}
    return {name: np.asarray([lookup[name][r[name]] for r in rows], dtype=np.int64) for name in lookup}


def _score_logits(logits, targets):
    """Pure-NumPy score helper, also used to verify the epoch-zero checkpoint path."""
    accuracy = {h: float(np.mean(logits[h].argmax(1) == targets[h])) for h in N.HEADS}
    return accuracy["action"] + 0.25 * accuracy["intent"], accuracy


def _replace_best(score: float, best_score: float) -> bool:
    return score > best_score


def _head_epoch(model, pooled, state, y, batch_size, device, seed):
    import torch

    # Keep the transformer and intent/flow predictors in eval mode; activate dropout only in trainable heads.
    model.encoder.eval()
    model.intent.eval()
    model.flow.eval()
    model.dropout.train()
    model.state_mlp.train()
    model.action_hidden.train()
    model.action.train()
    optimizer = _head_epoch.optimizer
    order = np.random.default_rng(seed).permutation(len(pooled))
    pooled_t = torch.as_tensor(pooled, dtype=torch.float32, device=device)
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.long, device=device)
    total = 0.0
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        p, s = pooled_t[idx], state_t[idx]
        # The base intent head is frozen and supplies predicted probabilities, never gold intent features.
        with torch.no_grad():
            dropped = model.dropout(p)
            intent_prob = torch.softmax(model.intent(dropped), dim=-1)
        state_embedding = model.state_mlp(s)
        action_input = torch.cat((dropped, intent_prob, state_embedding), dim=-1)
        action_logits = model.action(model.action_hidden(action_input))
        loss = torch.nn.functional.cross_entropy(action_logits, y_t[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(_head_epoch.trainable, 1.0)
        optimizer.step()
        total += float(loss.detach()) * len(idx)
    return total / len(order)


def _selection_score(model, pooled, state, rows, label_map, batch_size, device):
    import torch

    model.eval()
    labels = _labels(rows, label_map)
    outputs = {h: [] for h in N.HEADS}
    pooled_t = torch.as_tensor(pooled, dtype=torch.float32, device=device)
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            sl = slice(start, min(start + batch_size, len(rows)))
            logits = _head_logits(model, pooled_t[sl], state_t[sl])
            for h in N.HEADS:
                outputs[h].append(logits[h].float().cpu().numpy())
    logits = {h: np.concatenate(outputs[h], axis=0) for h in N.HEADS}
    return _score_logits(logits, labels)


def refine(args):
    import torch

    if args.tag == args.base_tag:
        raise ValueError("refined --tag must differ from --base-tag")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    N._seed_everything(SEED)
    torch.set_num_threads(args.threads)

    fit = [M.model_input(r) | {k: r[k] for k in ("intent", "flow", "action")} for r in M.partition("fit")]
    selection = [M.model_input(r) | {k: r[k] for k in ("intent", "flow", "action")}
                 for r in M.partition("selection")]
    labels = M.labels()
    model, tokenizer, metadata, _, _ = N._load_checkpoint(args.base_tag, args.device)
    if metadata["labels"] != labels:
        raise ValueError("base checkpoint class order does not match current fit labels")
    mean = np.asarray(metadata["state_mean"], dtype=np.float32)
    std = np.asarray(metadata["state_std"], dtype=np.float32)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    producer = _producer_identity(args.base_tag)
    cache_f, ids_f, pooled_f_path, state_f_path, meta_f = _cache_paths(args.base_tag, "fit")
    cache_s, ids_s, pooled_s_path, state_s_path, meta_s = _cache_paths(args.base_tag, "selection")
    cache_f.mkdir(parents=True, exist_ok=True)
    cache_s.mkdir(parents=True, exist_ok=True)

    # Build/cache pooled features and normalized observed-action state for only fit and selection.
    feature_sets = {}
    cache_status = {}
    for name, rows, ids_path, pooled_path, state_path, cache_meta_path in (
        ("fit", fit, ids_f, pooled_f_path, state_f_path, meta_f),
        ("selection", selection, ids_s, pooled_s_path, state_s_path, meta_s),
    ):
        ids = [r["id"] for r in rows]
        cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8")) if cache_meta_path.exists() else {}
        if (ids_path.exists() and pooled_path.exists() and state_path.exists()
                and cache_meta.get("producer_identity") == producer
                and json.loads(ids_path.read_text(encoding="utf-8")) == ids):
            pooled = np.load(pooled_path).astype(np.float32)
            state = np.load(state_path).astype(np.float32)
            if (pooled.shape != (len(rows), int(metadata["encoder_config"]["hidden_size"]))
                    or state.shape != (len(rows), int(metadata["state_dim"]))):
                raise ValueError(f"cached {name} feature shape mismatch")
            cache_status[name] = "reused"
            check_diff = None
        else:
            sequences = N.encode_rows(tokenizer, rows, metadata["max_length"])
            raw_state = N._state_norm(N._state_rows(rows), mean, std)
            pooled, check_diff = _pool_and_check(model, sequences, raw_state, rows,
                                                 args.feature_bs, pad_id, args.device)
            state = raw_state
            np.save(pooled_path, pooled)
            np.save(state_path, state)
            ids_path.write_text(json.dumps(ids) + "\n", encoding="utf-8")
            cache_meta_path.write_text(json.dumps({"producer_identity": producer,
                                                   "base_tag": args.base_tag,
                                                   "feature_version": FEATURE_VERSION,
                                                   "split": name}, indent=2) + "\n", encoding="utf-8")
            cache_status[name] = "built"
        feature_sets[name] = (pooled, state)
        cache_status[f"{name}_max_forward_diff"] = check_diff

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_modules = (model.state_mlp, model.action_hidden, model.action)
    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no action/state head parameters were made trainable")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    _head_epoch.optimizer = optimizer
    _head_epoch.trainable = trainable

    y = _labels(fit, labels)["action"]
    fit_features, fit_state = feature_sets["fit"]
    sel_features, sel_state = feature_sets["selection"]
    base_score, base_accuracy = _selection_score(model, sel_features, sel_state, selection,
                                                 labels, args.bs, args.device)
    best_score = base_score
    best_accuracy = dict(base_accuracy)
    best_epoch = 0
    best_metadata = dict(metadata)
    best_metadata.update(tag=args.tag, head_refinement={"base_tag": args.base_tag, "seed": SEED,
                                                         "epochs": 0, "lr": args.lr,
                                                         "batch_size": args.bs, "trainable": ["state_mlp", "action_hidden", "action"],
                                                         "feature_cache": str(cache_f)})
    _, checkpoint_dir = N._dirs(args.tag)
    N._save_best(model, tokenizer, model.encoder.config, checkpoint_dir, best_metadata, best_score)
    history = [{"epoch": 0, "train_loss": None, "selection_accuracy": base_accuracy,
                "selection_score": base_score}]

    torch.set_num_threads(args.head_threads)
    for epoch in range(1, args.epochs + 1):
        train_loss = _head_epoch(model, fit_features, fit_state, y, args.bs, args.device, SEED + epoch)
        score, accuracy = _selection_score(model, sel_features, sel_state, selection, labels, args.bs, args.device)
        entry = {"epoch": epoch, "train_loss": train_loss, "selection_accuracy": accuracy,
                 "selection_score": score}
        history.append(entry)
        print(json.dumps({"event": "head_epoch", **entry}, separators=(",", ":")), flush=True)
        if _replace_best(score, best_score):
            best_score = score
            best_accuracy = dict(accuracy)
            best_epoch = epoch
            best_metadata = dict(metadata)
            best_metadata.update(tag=args.tag, head_refinement={"base_tag": args.base_tag, "seed": SEED,
                                                                 "epochs": epoch, "lr": args.lr,
                                                                 "batch_size": args.bs,
                                                                 "trainable": ["state_mlp", "action_hidden", "action"],
                                                                 "feature_cache": str(cache_f)})
            N._save_best(model, tokenizer, model.encoder.config, checkpoint_dir, best_metadata, best_score)

    out_dir, _ = N._dirs(args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"base_tag": args.base_tag, "tag": args.tag, "seed": SEED, "device": args.device,
              "threads_for_feature_extraction": args.threads, "threads_for_head_training": args.head_threads,
              "epochs": args.epochs, "batch_size": args.bs, "lr": args.lr,
              "fit_rows": len(fit), "selection_rows": len(selection), "cache": cache_status,
              "base_selection_accuracy": base_accuracy, "base_selection_score": base_score,
              "best_selection_accuracy": best_accuracy,
              "base_action_accuracy": base_accuracy["action"],
              "best_action_accuracy": best_accuracy["action"],
              "action_accuracy_gain": best_accuracy["action"] - base_accuracy["action"],
              "minimum_action_gain_for_root_choice": 0.005,
              "best_epoch": best_epoch,
              "best_selection_score": best_score, "selection_score_gain": best_score - base_score,
              "selection_history": history, "calibration_or_confirmation_accessed": False}
    (out_dir / "head_refinement.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "head_refinement_complete", "base_score": base_score,
                      "best_score": best_score, "gain": best_score - base_score}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-tag", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--head-threads", type=int, default=4)
    parser.add_argument("--feature-bs", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if (args.threads < 1 or args.head_threads < 1 or args.feature_bs < 1
            or args.epochs < 1 or args.bs < 1 or args.lr <= 0):
        raise SystemExit("threads, epochs, batch size and learning rate must be positive")
    refine(args)


if __name__ == "__main__":
    main()
