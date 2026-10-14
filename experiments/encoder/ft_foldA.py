"""Fold-A fine-tune of a small encoder on CPU (intent 96 + flow 10 + next-action 30 heads), for a fair comparison
with the shipped system on train fold B. Copied from the 13 Oct probe's ft.py; changes: trains on fold A only
(the same seed-0 split as triage.classical.folds(), ids in folds_ids.json), early-stops on 10% of fold A by
conversation hash, and writes probabilities for fold B, dev, dev_noisy and hostile. Runs on the remote CPU box.
The 14 Oct run's cache/ and logs/ files for tag ett32A were copied into results/raw/encoder-ett32A/.

Original docstring follows.
B3: fine-tune a small encoder on CPU with intent (96) + flow (10) + next-action (30) heads.

    python ft.py --model microsoft/deberta-v3-small --tag deb --threads 24 --epochs 3 --max-len 256
    python ft.py --model answerdotai/ModernBERT-base --tag mbert --bench   # time 12 steps only

Input: speaker-tagged context (customer/agent/action), each turn NFKC-normalized and capped to its
first 400 chars, then token-truncated from the LEFT so the most recent turns are kept.
Train: 90% of train (split by conversation); the other 10% is used for early stopping and for
temperature scaling. dev / dev_noisy are scored only, never fitted on.
Writes cache/ft_<tag>__<head>__<split>.npy (probabilities, temperature-scaled) and logs/ft_<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

import common

HEADS = ("intent", "flow", "action")
LOSS_W = {"intent": 1.0, "flow": 0.5, "action": 1.0}


def encode_rows(tok, rows, max_len):
    """Token ids per row: turns tokenized separately, most recent turns kept, special tokens added."""
    full = tok("hello", add_special_tokens=True)["input_ids"]
    inner = tok("hello", add_special_tokens=False)["input_ids"]
    st = next(i for i in range(len(full)) if full[i:i + len(inner)] == inner)
    prefix, suffix = full[:st], full[st + len(inner):]
    budget = max_len - len(prefix) - len(suffix)
    out = []
    for r in rows:
        ts = [(s, t) for s, t in common.turns(r) if t]
        pieces = [tok(f"{s}: {t}\n", add_special_tokens=False)["input_ids"] for s, t in ts]
        keep, total = [], 0
        for p in reversed(pieces):
            if total + len(p) > budget:
                if not keep:  # a single over-long last turn: keep its head
                    keep.append(p[:budget])
                break
            keep.append(p)
            total += len(p)
        ids = [i for p in reversed(keep) for i in p]
        out.append(prefix + ids + suffix)
    return out


class Model(nn.Module):
    def __init__(self, name, n_out: dict):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(name, dtype=torch.float32)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({k: nn.Linear(h, v) for k, v in n_out.items()})

    def forward(self, ids, mask):
        hs = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(hs.dtype)
        pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1.0)  # mean pooling
        pooled = self.drop(pooled)
        return {k: head(pooled) for k, head in self.heads.items()}


def batches(seqs, idx, bs, pad_id, shuffle):
    idx = list(idx)
    if shuffle:  # length bucketing: shuffle, sort within chunks of 50 batches, shuffle batches
        random.shuffle(idx)
        chunks = [sorted(idx[i:i + bs * 50], key=lambda j: len(seqs[j])) for i in range(0, len(idx), bs * 50)]
        bl = [c[i:i + bs] for c in chunks for i in range(0, len(c), bs)]
        random.shuffle(bl)
    else:
        idx = sorted(idx, key=lambda j: len(seqs[j]))
        bl = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    for b in bl:
        L = max(len(seqs[j]) for j in b)
        ids = torch.full((len(b), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(b), L), dtype=torch.long)
        for k, j in enumerate(b):
            ids[k, :len(seqs[j])] = torch.tensor(seqs[j])
            mask[k, :len(seqs[j])] = 1
        yield b, ids, mask


@torch.no_grad()
def predict_logits(model, seqs, idx, pad_id, bs=64):
    model.eval()
    out = {k: np.zeros((len(idx), model.heads[k].out_features), dtype=np.float32) for k in HEADS}
    pos = {j: n for n, j in enumerate(idx)}
    for b, ids, mask in batches(seqs, idx, bs, pad_id, shuffle=False):
        lo = model(ids, mask)
        for k in HEADS:
            out[k][[pos[j] for j in b]] = lo[k].float().numpy()
    return out


def fit_temperature(logits, y):
    lo = torch.tensor(logits)
    yt = torch.tensor(y)
    best = (1.0, 1e9)
    for T in np.arange(0.5, 3.01, 0.05):
        nll = Fn.cross_entropy(lo / T, yt).item()
        if nll < best[1]:
            best = (float(T), nll)
    return best[0]


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--evals-per-epoch", type=int, default=2)
    ap.add_argument("--time-budget-min", type=float, default=60)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--freeze-emb", action="store_true",
                    help="freeze the input word-embedding matrix (big-vocab models: saves the dense AdamW update)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    from transformers import AutoTokenizer
    t0 = time.time()
    train_all, dev, noisy, hostile = common.load_all()
    fold = json.load(open(common.WORK / "folds_ids.json"))
    by_id = {r["id"]: r for r in train_all}
    train = [by_id[i] for i in fold["A"]]
    foldB = [by_id[i] for i in fold["B"]]
    labels = {"intent": sorted({r["intent"] for r in train}), "flow": sorted({r["flow"] for r in train}),
              "action": sorted({r["action"] for r in train})}
    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    splits = {"train": train, "foldB": foldB, "dev": dev, "noisy": noisy, "hostile": hostile}
    seqs = {s: encode_rows(tok, rows, args.max_len) for s, rows in splits.items()}
    lens = np.array([len(x) for x in seqs["train"]])
    print(f"tokenized in {time.time() - t0:.0f}s; train len mean {lens.mean():.0f} p95 {np.percentile(lens, 95):.0f} "
          f"max {lens.max()} at cap {args.max_len}: {(lens >= args.max_len).mean():.3f}", flush=True)

    mask = np.array(common.holdout_mask(train, 0.10))
    tr_idx = np.where(~mask)[0]
    ho_idx = np.where(mask)[0]
    Y = {k: np.array([labels[k].index(r[k]) for r in train]) for k in HEADS}

    model = Model(args.model, {k: len(v) for k, v in labels.items()})
    n_params = sum(p.numel() for p in model.parameters())
    if args.freeze_emb:
        model.enc.get_input_embeddings().weight.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(tr_idx) / args.bs)
    total_steps = int(steps_per_epoch * args.epochs)
    warm = int(0.06 * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else max(0.0, (total_steps - s) / max(1, total_steps - warm)))

    log = {"model": args.model, "tag": args.tag, "threads": args.threads, "max_len": args.max_len,
           "bs": args.bs, "lr": args.lr, "epochs_planned": args.epochs, "n_params": n_params,
           "fp32_mb": round(n_params * 4 / 2**20, 1),
           "trainable_params": sum(p.numel() for p in trainable), "freeze_emb": args.freeze_emb, "train_rows": int(len(tr_idx)), "holdout_rows": int(len(ho_idx)),
           "len_mean": float(lens.mean()), "len_p95": float(np.percentile(lens, 95)), "evals": []}

    eval_every = max(1, steps_per_epoch // args.evals_per_epoch)
    step, best, best_state = 0, -1.0, None
    t_train = time.time()
    stop = False
    ep = 0
    while not stop:
        model.train()
        for b, ids, msk in batches(seqs["train"], tr_idx, args.bs, pad_id, shuffle=True):
            lo = model(ids, msk)
            loss = sum(LOSS_W[k] * Fn.cross_entropy(lo[k], torch.tensor(Y[k][b])) for k in HEADS)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 20 == 0 or (args.bench and step <= 12):
                el = time.time() - t_train
                print(f"step {step}/{total_steps} loss {loss.item():.3f} {el / step:.2f}s/step "
                      f"eta_total {el / step * total_steps / 60:.1f}min", flush=True)
            if args.bench and step >= 12:
                el = (time.time() - t_train) / step
                print(json.dumps({"bench": args.model, "threads": args.threads, "s_per_step": round(el, 2),
                                  "est_epoch_min": round(el * steps_per_epoch / 60, 1)}), flush=True)
                return
            over_budget = (time.time() - t_train) / 60 > args.time_budget_min
            if step % eval_every == 0 or step >= total_steps or over_budget:
                L = predict_logits(model, seqs["train"], ho_idx, pad_id)
                acc = {k: float((L[k].argmax(1) == Y[k][ho_idx]).mean()) for k in HEADS}
                score = acc["intent"] + acc["action"]
                ev = {"step": step, "epoch": round(step / steps_per_epoch, 2),
                      "min": round((time.time() - t_train) / 60, 1), **{f"ho_{k}": round(v, 4) for k, v in acc.items()}}
                log["evals"].append(ev)
                print("EVAL", json.dumps(ev), flush=True)
                if score > best:
                    best = score
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    log["best_eval"] = ev
                model.train()
                if over_budget:
                    log["stopped_on_time_budget"] = True
                    stop = True
                    break
            if step >= total_steps:
                stop = True
                break
        ep += 1
    log["train_min"] = round((time.time() - t_train) / 60, 1)
    log["steps"] = step
    model.load_state_dict(best_state)

    # temperature per head on the train holdout
    Lho = predict_logits(model, seqs["train"], ho_idx, pad_id)
    temps = {k: fit_temperature(Lho[k], Y[k][ho_idx]) for k in HEADS}
    log["temperature"] = temps
    res = {}
    torch.save({k: v.half() for k, v in best_state.items()}, common.WORK / "models" / f"ft_{args.tag}.pt")
    for s in ("foldB", "dev", "noisy", "hostile"):
        (common.CACHE / f"ft_{args.tag}__ids__{s}.json").write_text(json.dumps([r["id"] for r in splits[s]]))
        L = predict_logits(model, seqs[s], list(range(len(seqs[s]))), pad_id)
        for k in HEADS:
            P = softmax(L[k] / temps[k])
            np.save(common.CACHE / f"ft_{args.tag}__{k}__{s}.npy", P.astype(np.float32))
            if s == "hostile":
                continue
            P_raw = softmax(L[k])
            gold = [r[k] for r in splits[s]]
            pred = [labels[k][i] for i in P.argmax(1)]
            ok = [g == p for g, p in zip(gold, pred)]
            res[f"{k}_{s}_acc"] = round(float(np.mean(ok)), 4)
            res[f"{k}_{s}_ece_raw"] = round(common.ece(P_raw.max(1), ok), 4)
            res[f"{k}_{s}_ece_T"] = round(common.ece(P.max(1), ok), 4)
            if k == "intent":
                res[f"intent_{s}_f1"] = round(common.macro_f1(gold, pred), 4)
            if s == "noisy":
                res[f"{k}_{s}_by_kind"] = common.per_kind(noisy, ok)
    log["results"] = res
    log["hostile_ok"] = True  # reached here = all 9 hostile rows tokenized and encoded without exception
    print(json.dumps(res), flush=True)

    # CPU single-request latency (tokenize + encode + heads), batch size 1, on 200 dev rows
    model.eval()
    rng = random.Random(0)
    sample = rng.sample(dev, 200)
    lat = {}
    for th in (1, 2, 4):
        torch.set_num_threads(th)
        ts = []
        with torch.no_grad():
            for i, r in enumerate(sample):
                t = time.perf_counter()
                ids = encode_rows(tok, [r], args.max_len)[0]
                x = torch.tensor([ids])
                model(x, torch.ones_like(x))
                if i >= 10:  # skip warm-up
                    ts.append((time.perf_counter() - t) * 1000)
        lat[f"threads_{th}"] = {"p50_ms": round(float(np.percentile(ts, 50)), 1),
                                "p95_ms": round(float(np.percentile(ts, 95)), 1)}
        print("latency", th, lat[f"threads_{th}"], flush=True)
    log["latency_bs1"] = lat
    log["total_min"] = round((time.time() - t0) / 60, 1)
    (common.LOGS / f"ft_{args.tag}.json").write_text(json.dumps(log, indent=2))
    with open(common.LOGS / "labels.json", "w") as fh:
        json.dump(labels, fh)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
