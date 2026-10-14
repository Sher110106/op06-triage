"""Data loading and the input projection. Models only ever see what `project()` returns."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "op06"
SPLITS = {"train": "train.jsonl.gz", "dev": "dev.jsonl", "noisy": "dev_noisy.jsonl", "hostile": "hostile.jsonl"}


def load_jsonl(path) -> list[dict]:
    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_split(name: str) -> list[dict]:
    return load_jsonl(DATA / SPLITS[name])


def project(row: dict) -> dict:
    """Same allowlist as data/op06/evaluation_inputs.py for OP-06: id, turn_index, context[speaker, text]."""
    out = {k: row[k] for k in ("id", "turn_index") if k in row}
    out["context"] = [{k: t[k] for k in ("speaker", "text") if k in t} for t in row["context"]]
    return out


def conv_id(rid: str) -> str:
    """'3129-8' and '3129-8-noisy' both belong to conversation '3129'."""
    return str(rid).removesuffix("-noisy").rsplit("-", 1)[0]
