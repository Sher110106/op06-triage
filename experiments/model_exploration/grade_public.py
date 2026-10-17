"""Join all three public slices and run the unchanged official grader; no model inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.model_exploration import model_common as M
from harness.eval import G
from triage.data import DATA


def grade(tag: str) -> dict:
    target = M.OUT / tag
    schema = json.loads((DATA / "schema.json").read_text())
    gold = [M.partition(s) for s in ("dev", "noisy", "hostile")]
    expected = [r["id"] for rows in gold for r in rows]
    summaries = {}
    for name in ("reference", "encoder", "ensemble"):
        predictions = []
        for split in ("dev", "noisy", "hostile"):
            path = M.OUT / f"{name}_{split}.jsonl"
            predictions.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        ids = [p["id"] for p in predictions]
        if ids != expected or len(set(ids)) != len(ids):
            raise ValueError(f"{name} public predictions do not align with every expected input")
        joined = target / f"{name}_public.jsonl"
        joined.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions))
        report = G.grade(*gold, {p["id"]: p for p in predictions}, schema)
        M.write_json(target / f"{name}_public_official.json", report)
        summaries[name] = {"rows": len(predictions), "intent_accuracy": report["intent"]["accuracy_all"],
                           "macro_f1": report["intent"]["macro_f1_all"],
                           "action_accuracy": report["action"]["accuracy_all"],
                           "clean_action_accuracy": report["action"]["accuracy_clean"],
                           "noisy_action_accuracy": report["action"]["accuracy_noisy"],
                           "noisy_gap": report["action"]["noisy_gap"],
                           "schema_valid_rate": report["schema"]["valid_rate_all_inputs"],
                           "top70_error_ratio": report["confidence"]["top70_error_ratio"],
                           "bars": {k: v["pass"] for k, v in report["bars"].items()}}
    M.write_json(target / "public_official_summary.json", summaries)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    print(json.dumps(grade(args.tag), indent=2))
