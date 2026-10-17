"""Compare frozen replay and real inference on the first selection rows, without scoring labels.

python experiments/model_exploration/verify_runtime.py --tag modernbert_joint --limit 100 --threads 2

Runs no training or provider calls. Reports parity only; confirmation and development partitions
are never requested. The optional experiment service wrapper is imported only when this CLI runs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.model_exploration import evaluate as E, model_common as M, neural as N
from triage import classical as C
from triage.data import project
from triage.policy import DEFAULT_POLICY, load as load_policy

CONFIDENCE_TOLERANCE = 1e-5
EXACT_FIELDS = ("id", "intent", "action", "needs_human")


def _compare(expected: dict, actual: dict, comparison: str, row_id: str) -> tuple[list[dict], float]:
    failures = []
    for field in EXACT_FIELDS:
        if actual.get(field) != expected.get(field) or type(actual.get(field)) is not type(expected.get(field)):
            failures.append({"comparison": comparison, "id": row_id, "field": field,
                             "expected": expected.get(field), "actual": actual.get(field)})
    expected_conf, actual_conf = expected.get("confidence"), actual.get("confidence")
    valid = all(isinstance(value, (float, int)) and not isinstance(value, bool)
                and math.isfinite(value) for value in (expected_conf, actual_conf))
    difference = abs(float(expected_conf) - float(actual_conf)) if valid else None
    if difference is None or difference > CONFIDENCE_TOLERANCE:
        failures.append({"comparison": comparison, "id": row_id, "field": "confidence",
                         "expected": expected_conf, "actual": actual_conf, "absolute_diff": difference})
    return failures, difference if difference is not None else 0.0


def _cached_predictions(tag: str, limit: int):
    rows_all = M.partition("selection")
    ids_all = [row["id"] for row in rows_all]
    if len(set(ids_all)) != len(ids_all):
        raise ValueError("selection IDs are not unique")
    rows = rows_all[:limit]
    if not rows:
        raise ValueError("selection partition is empty")
    frozen = json.loads((M.OUT / tag / "frozen.json").read_text())
    if frozen["tag"] != tag:
        raise ValueError("frozen ensemble tag does not match --tag")
    policy = load_policy(DEFAULT_POLICY)
    if policy.version != frozen["policy_version"]:
        raise ValueError("current policy version differs from the frozen comparison")
    # Do not silently generate missing reference exports during a parity check.
    for filename in ("probs_selection.npz", "ids_selection.json"):
        path = M.OUT / "reference" / filename
        if not path.exists():
            raise FileNotFoundError(f"frozen reference export missing: {path}")
    reference = E.reference("selection")
    logits = E.neural_logits(tag, "selection")
    for name, values in (("reference", reference), ("neural", logits)):
        for head in E.HEADS:
            expected_shape = (len(rows_all), len(M.labels()[head]))
            if values[head].shape != expected_shape or not np.isfinite(values[head]).all():
                raise ValueError(f"invalid {name} {head} selection array: expected {expected_shape}")
    reference = {head: value[:len(rows)] for head, value in reference.items()}
    neural = {head: C.softmax(logits[head][:len(rows)], frozen["temperatures"][head]) for head in E.HEADS}
    raw = E.mix(reference, neural, frozen["classical_action_flow_weight"], frozen["classical_intent_weight"])
    fitted = joblib.load(M.MODEL_DIR / f"{tag}-calibration.joblib")["ensemble"]
    if fitted["thresholds"] != frozen["thresholds"]["ensemble"]:
        raise ValueError("saved ensemble cutoffs differ from frozen settings")
    predictions = E.predict(raw, rows, fitted["clf"], fitted["thresholds"], policy)
    if set(predictions) != {row["id"] for row in rows}:
        raise ValueError("cached replay IDs differ from the selected prefix")
    return rows, predictions


def verify(tag: str, limit: int, threads: int) -> dict:
    import torch
    from experiments.model_exploration.runtime import NeuralEnsemblePredictor
    from experiments.model_exploration.serve import NeuralTriage

    rows, expected = _cached_predictions(tag, limit)
    projected = [project(row) for row in rows]
    report = {"tag": tag, "partition": "selection", "selection_rule": "first rows in saved partition order",
              "rows": len(rows), "singleton_rows": min(10, len(rows)), "threads": threads,
              "confidence_tolerance": CONFIDENCE_TOLERANCE, "ids": [row["id"] for row in rows],
              "label_accuracy_scored": False, "llm_enabled": False,
              "max_confidence_diff": {"cached_vs_batch": 0.0, "cached_vs_singleton": 0.0,
                                      "batch_vs_singleton": 0.0},
              "mismatches": [], "error_traces": [], "pass": False}
    predictor = NeuralEnsemblePredictor(tag, threads=threads, device="cpu", batch_size=32)
    runner = None
    try:
        runner = NeuralTriage(predictor)
        report["torch_threads_on_request_worker"] = runner._on_cpu(torch.get_num_threads)
        if report["torch_threads_on_request_worker"] != threads:
            raise RuntimeError("request worker does not use the requested Torch thread count")
        if runner.llm is not None:
            raise RuntimeError("runtime parity requires the LLM to be disabled")
        report["runtime"] = predictor.metadata
        outputs = runner.handle_many(projected, [f"parity-batch-{i}" for i in range(len(rows))])
        if len(outputs) != len(rows):
            raise RuntimeError(f"batch returned {len(outputs)} responses for {len(rows)} rows")
        batch = {}
        for row, (response, trace) in zip(rows, outputs):
            rid = row["id"]
            batch[rid] = response
            if trace.get("error"):
                report["error_traces"].append({"mode": "batch", "id": rid, "response": response, "trace": trace})
            failures, difference = _compare(expected[rid], response, "cached_vs_batch", rid)
            report["mismatches"].extend(failures)
            report["max_confidence_diff"]["cached_vs_batch"] = max(
                report["max_confidence_diff"]["cached_vs_batch"], difference)
        for i, row in enumerate(projected[:report["singleton_rows"]]):
            rid = row["id"]
            response, trace = runner.handle(row, f"parity-singleton-{i}")
            if trace.get("error"):
                report["error_traces"].append({"mode": "singleton", "id": rid, "response": response, "trace": trace})
            for comparison, baseline in (("cached_vs_singleton", expected[rid]), ("batch_vs_singleton", batch[rid])):
                failures, difference = _compare(baseline, response, comparison, rid)
                report["mismatches"].extend(failures)
                report["max_confidence_diff"][comparison] = max(report["max_confidence_diff"][comparison], difference)
        report["pass"] = not report["mismatches"] and not report["error_traces"]
    except Exception as exc:
        report["exception"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if runner is not None:
            runner.close()
        else:
            predictor.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.limit < 1 or args.threads < 1:
        parser.error("--limit and --threads must be positive")
    output_dir, _ = N._dirs(args.tag)
    try:
        report = verify(args.tag, args.limit, args.threads)
    except Exception as exc:
        report = {"tag": args.tag, "partition": "selection", "pass": False,
                  "label_accuracy_scored": False, "exception": {"type": type(exc).__name__, "message": str(exc)}}
    path = output_dir / "runtime_parity.json"
    M.write_json(path, report)
    print(json.dumps({"report": str(path), **report}, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
