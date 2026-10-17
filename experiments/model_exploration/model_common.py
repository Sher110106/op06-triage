"""Shared, conversation-grouped data boundaries for the stronger-model experiment."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triage.data import conv_id, load_split, project  # noqa: E402
from triage import gate  # noqa: E402

OUT = ROOT / "results" / "model-exploration-20261017"
MODEL_DIR = ROOT / "models" / "model-exploration"


@lru_cache(maxsize=1)
def _partitions():
    rows = load_split("train")
    groups = np.asarray([conv_id(r["id"]) for r in rows])
    remaining, confirmation = next(GroupShuffleSplit(
        n_splits=1, test_size=.15, random_state=20261020).split(rows, groups=groups))
    rest, cal = next(GroupShuffleSplit(
        n_splits=1, test_size=.10 / .85, random_state=20261021
    ).split(remaining, groups=groups[remaining]))
    calibration = remaining[cal]
    remaining = remaining[rest]
    fit, sel = next(GroupShuffleSplit(
        n_splits=1, test_size=.10 / .75, random_state=20261022
    ).split(remaining, groups=groups[remaining]))
    indices = {"fit": remaining[fit], "selection": remaining[sel],
               "calibration": calibration, "confirmation": confirmation}
    grouped = {k: set(groups[v]) for k, v in indices.items()}
    assert sum(map(len, indices.values())) == len(rows)
    names = list(grouped)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            assert grouped[name].isdisjoint(grouped[other]), (name, other)
    return {k: [rows[i] for i in idx] for k, idx in indices.items()}


def partition(name):
    if name in ("dev", "noisy", "hostile"):
        return load_split(name)
    return _partitions()[name]


def model_input(row):
    """Use the same observed-only gate-cleaned input for every model family."""
    result = gate.clean(project(row))
    return {"id": result.id, "context": result.context, "turn_index": result.turn_index}


@lru_cache(maxsize=1)
def labels():
    fit = partition("fit")
    result = {head: sorted({r[head] for r in fit}) for head in ("intent", "flow", "action")}
    assert {k: len(v) for k, v in result.items()} == {"intent": 96, "flow": 10, "action": 30}
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1) + "\n")


def record_partitions():
    parts = _partitions()
    record = {"seeds": [20261020, 20261021, 20261022],
              "note": "New grouped re-split of previously studied public training data; not pristine or hidden test.",
              "parts": {name: {"rows": len(rows), "conversations": len({conv_id(r['id']) for r in rows}),
                                 "ids": [r["id"] for r in rows]} for name, rows in parts.items()}}
    write_json(OUT / "partitions.json", record)
    return {name: {k: v for k, v in item.items() if k != "ids"}
            for name, item in record["parts"].items()}


if __name__ == "__main__":
    print(json.dumps({"partitions": record_partitions(),
                      "labels": {k: len(v) for k, v in labels().items()}}))
