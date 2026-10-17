"""Offline reports for complete, explicit hosted-model runs; this script never contacts a provider.

Selection choice:
  python experiments/model_exploration/analyze_hosted.py select \
      --runs selection-luna selection-haiku selection-flash-low

An explicitly requested cohort comparison (for example, after freezing a winner):
  python experiments/model_exploration/analyze_hosted.py compare \
      --cohort results/model-exploration-20261017/cohorts/confirmation.json \
      --runs confirmation-luna confirmation-haiku

Both commands require complete attempts for every frozen cohort item and full-split predictions that preserve
CPU fallbacks. Selection writes a frozen choice file. No confirmation data is read unless `compare` receives an
explicit confirmation cohort path.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.model_exploration import model_common as M  # noqa: E402
from triage import prompts as P  # noqa: E402

OUT = Path(M.OUT)
HOSTED = OUT / "hosted"
EXPECTED_ATTEMPTS = 300
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20261024
SELECTION_RUNS = ("selection-luna", "selection-haiku", "selection-flash-low")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read {path}: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read {path}: {type(exc).__name__}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise SystemExit(f"Expected JSON objects in {path}")
    return rows


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finite(v):
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v)) and float(v) >= 0)


def _numeric_summary(records: list[dict], key: str) -> dict:
    values = [float(r[key]) for r in records if _finite(r.get(key))]
    return {"known_subtotal": float(sum(values)), "known_count": len(values),
            "missing_count": len(records) - len(values),
            "complete_total": float(sum(values)) if len(values) == len(records) else None}


def _count(values) -> dict:
    return dict(sorted(Counter("unknown" if v is None else str(v) for v in values).items()))


def _parsed_answer(call: dict) -> tuple[dict | None, str]:
    response = call.get("response")
    choices = response.get("choices") if isinstance(response, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    return P.parse_output(content if isinstance(content, str) else None)


def _resolve_repo_path(value: str, name: str) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise SystemExit(f"{name} path leaves the repository")
    return path


def _load_cohort(path_value: str) -> tuple[dict, str, Path]:
    path = _resolve_repo_path(path_value, "cohort")
    raw = path.read_bytes()
    cohort = json.loads(raw)
    if not isinstance(cohort, dict) or cohort.get("format") != 1 or not isinstance(cohort.get("items"), list):
        raise SystemExit(f"Unsupported or malformed frozen cohort: {path}")
    return cohort, _sha256(raw), path


def _unique_rows(rows: list[dict], name: str) -> dict[str, dict]:
    ids = [r.get("id") for r in rows]
    if any(not isinstance(i, str) for i in ids) or len(ids) != len(set(ids)):
        raise SystemExit(f"{name} IDs must be unique strings")
    return dict(zip(ids, rows))


def _check_output_shape(rows: list[dict], expected_ids: list[str], name: str) -> dict[str, dict]:
    ids = [r.get("id") for r in rows]
    if ids != expected_ids:
        raise SystemExit(f"{name} IDs/order do not exactly match the frozen partition")
    return _unique_rows(rows, name)


def _load_run(run_id: str, cohort: dict, cohort_hash: str, cohort_path: Path,
              split_rows: list[dict], fallback: dict[str, dict], items_by_id: dict[str, dict]) -> dict:
    run_dir = HOSTED / run_id
    run_meta_path, summary_path = run_dir / "run.json", run_dir / "summary.json"
    meta, summary = _read_json(run_meta_path), _read_json(summary_path)
    for name, value in (("run.json", meta), ("summary.json", summary)):
        if value.get("cohort_sha256") != cohort_hash:
            raise SystemExit(f"{run_id}/{name} cohort digest does not match {cohort_path}")
        if value.get("run_id", run_id) != run_id:
            raise SystemExit(f"{run_id}/{name} run ID mismatch")
        if value.get("model") != meta.get("model"):
            raise SystemExit(f"{run_id} model ID differs between run.json and summary.json")
    if meta.get("split") != cohort.get("split"):
        raise SystemExit(f"{run_id} split {meta.get('split')!r} does not match cohort {cohort.get('split')!r}")
    if summary.get("selected_rows") != EXPECTED_ATTEMPTS or summary.get("attempted_rows") != EXPECTED_ATTEMPTS:
        raise SystemExit(f"{run_id} is incomplete: expected {EXPECTED_ATTEMPTS} selected and attempted rows")
    if len(items_by_id) != EXPECTED_ATTEMPTS:
        raise SystemExit(f"cohort must contain exactly {EXPECTED_ATTEMPTS} unique items")

    calls_path = _resolve_repo_path(summary.get("raw_calls", f"results/model-exploration-20261017/hosted/{run_id}/raw_calls.jsonl"),
                                    f"{run_id} raw calls")
    calls = _read_jsonl(calls_path)
    call_map = _unique_rows(calls, f"{run_id} attempts")
    if set(call_map) != set(items_by_id) or len(calls) != EXPECTED_ATTEMPTS:
        raise SystemExit(f"{run_id} must have exactly one finalized attempt per frozen cohort ID")
    if any(r.get("model_requested") != meta.get("model") for r in calls):
        raise SystemExit(f"{run_id} contains an attempt for a different requested model")
    started_path = run_dir / "started_calls.jsonl"
    started = _read_jsonl(started_path)
    started_ids = [r.get("id") for r in started]
    if len(started_ids) != EXPECTED_ATTEMPTS or set(started_ids) != set(items_by_id):
        raise SystemExit(f"{run_id} does not have one persisted start/reservation for every cohort ID")

    pred_path = _resolve_repo_path(summary.get("predictions", f"results/model-exploration-20261017/hosted/{run_id}/predictions_{cohort['split']}.jsonl"),
                                   f"{run_id} predictions")
    predictions = _read_jsonl(pred_path)
    full_ids = [str(r["id"]) for r in split_rows]
    pred_map = _check_output_shape(predictions, full_ids, f"{run_id} full-split predictions")
    fallback_map = fallback
    cohort_ids = set(items_by_id)
    for row_id, call in call_map.items():
        baseline = fallback_map[row_id]
        actual = pred_map[row_id]
        changed = bool(call.get("accepted_action") and call.get("action_changed"))
        selected = call.get("action_selected")
        recorded_changed = selected != baseline["action"] if call.get("accepted_action") else False
        if bool(call.get("action_changed")) != recorded_changed:
            raise SystemExit(f"{run_id} attempt changed-action flag is inconsistent for {row_id}")
        expected_prediction = dict(baseline)
        if changed:
            if not isinstance(selected, str):
                raise SystemExit(f"{run_id} accepted row {row_id} has no selected action")
            expected_prediction["action"] = selected
            expected_prediction["needs_human"] = True
        if actual != expected_prediction:
            raise SystemExit(f"{run_id} prediction for attempted row {row_id} does not match attempt/fallback semantics")
        item = items_by_id[row_id]
        if baseline.get("action") != item.get("reference", {}).get("action"):
            raise SystemExit(f"CPU fallback and frozen cohort reference disagree for {row_id}")
    for row_id in full_ids:
        if row_id not in cohort_ids and pred_map[row_id] != fallback_map[row_id]:
            raise SystemExit(f"{run_id} changed an unqueried row {row_id}")

    return {"run_id": run_id, "model": meta.get("model"), "meta": meta, "summary": summary,
            "calls": calls, "call_map": call_map, "predictions": pred_map, "calls_path": calls_path,
            "predictions_path": pred_path, "started": started}


def _paired_accuracy(rows: list[dict], baseline: dict[str, dict], candidate: dict[str, dict],
                     run_id: str, cohort_label: str, draws: int = BOOTSTRAP_DRAWS) -> dict:
    if not rows:
        raise SystemExit(f"Cannot score empty {cohort_label} rows")
    ids = [str(r["id"]) for r in rows]
    base_correct = np.asarray([baseline[i]["action"] == r["action"] for i, r in zip(ids, rows)], dtype=np.float64)
    new_correct = np.asarray([candidate[i]["action"] == r["action"] for i, r in zip(ids, rows)], dtype=np.float64)
    groups = [M.conv_id(i) for i in ids]
    unique = sorted(set(groups))
    group_ix = {g: j for j, g in enumerate(unique)}
    n = np.zeros(len(unique), dtype=np.float64)
    base_sum = np.zeros_like(n)
    new_sum = np.zeros_like(n)
    for g, b, c in zip(groups, base_correct, new_correct):
        j = group_ix[g]
        n[j] += 1
        base_sum[j] += b
        new_sum[j] += c
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = rng.multinomial(len(unique), np.full(len(unique), 1. / len(unique)), size=draws)
    den = weights @ n
    base_draw = (weights @ base_sum) / den
    new_draw = (weights @ new_sum) / den
    diff_draw = new_draw - base_draw
    base_acc, new_acc = float(base_correct.mean()), float(new_correct.mean())
    ci = np.quantile(diff_draw, [.025, .975]).tolist()
    diff = new_acc - base_acc
    return {"rows": len(rows), "conversations": len(unique), "accuracy_unit": "fraction (1.0 = 100%)",
            "base_action_accuracy": base_acc,
            "hosted_action_accuracy": new_acc, "accuracy_diff": new_acc - base_acc,
            "accuracy_diff_pp": 100.0 * diff,
            "paired_conversation_bootstrap_95ci_diff": ci,
            "paired_conversation_bootstrap_95ci_diff_pp": (100.0 * np.asarray(ci)).tolist()}


def _choose_selection(reports: list[dict]) -> tuple[dict, bool, list[dict]]:
    """Use gateway cost for an accuracy tie only when every tied candidate has a complete bill."""
    if not reports:
        raise ValueError("no selection candidates")
    best_accuracy = max(r["selected_tail"]["hosted_action_accuracy"] for r in reports)
    tied = [r for r in reports if r["selected_tail"]["hosted_action_accuracy"] == best_accuracy]
    bills = {r["run_id"]: r["costs_usd"]["gateway_reported"]["complete_total"] for r in tied}
    if any(value is None for value in bills.values()):
        preferred = next((r for r in tied if r["run_id"] == "selection-luna"), None)
        return preferred or min(tied, key=lambda r: r["run_id"]), False, tied
    return min(tied, key=lambda r: (bills[r["run_id"]], r["run_id"])), True, tied


def _deadline_replay(fallback: dict[str, dict], call_map: dict[str, dict]) -> tuple[dict[str, dict], dict]:
    """Illustrative action-only 4s gate: use accepted actions only when their recorded call fit the cutoff."""
    replay = {rid: dict(pred) for rid, pred in fallback.items()}
    accepted_in_time = changed_in_time = 0
    for rid, call in call_map.items():
        latency = call.get("latency_s")
        if not call.get("accepted_action") or not _finite(latency) or float(latency) > 4.0:
            continue
        accepted_in_time += 1
        selected = call.get("action_selected")
        if not isinstance(selected, str):
            continue
        if selected != replay[rid]["action"]:
            changed_in_time += 1
            replay[rid]["action"] = selected
            replay[rid]["needs_human"] = True
    counts = {"accepted_actions_within_4s": accepted_in_time,
              "changed_actions_within_4s": changed_in_time,
              "latency_source": "raw_calls.jsonl latency_s; excludes post-call meter refresh",
              "interpretation": "offline retrospective action replay; not service latency, provider cancellation, or actual cost"}
    return replay, counts


def _run_metrics(run: dict, cohort: dict, split_rows: list[dict], fallback: dict[str, dict]) -> dict:
    items = cohort["items"]
    tail_rows_by_id = {str(r["id"]): r for r in split_rows}
    tail_rows = [tail_rows_by_id[item["id"]] for item in items]
    tail_ids = [str(r["id"]) for r in tail_rows]
    calls = run["calls"]
    latencies = [float(r["latency_s"]) for r in calls if _finite(r.get("latency_s"))]
    attempts = len(calls)
    gateway = _numeric_summary(calls, "gateway_reported_cost_usd")
    list_cost = _numeric_summary(calls, "cost_list_uncached_usd")
    reservations = [float(r["reserved_usd"]) for r in run["started"] if _finite(r.get("reserved_usd"))]
    liabilities = [float(r["liability_usd"]) for r in calls if _finite(r.get("liability_usd"))]
    token_sums = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        vals = [r.get("usage", {}).get(name) if isinstance(r.get("usage"), dict) else None for r in calls]
        good = [int(v) for v in vals if isinstance(v, int) and not isinstance(v, bool) and v >= 0]
        token_sums[name] = {"sum_known": sum(good), "known_count": len(good), "missing_count": len(vals) - len(good)}
    statuses = [r.get("status") for r in calls]
    accepted = sum(bool(r.get("accepted_action")) for r in calls)
    changed = sum(bool(r.get("accepted_action") and r.get("action_changed")) for r in calls)
    errors = [r.get("error_class") or (r.get("status") if not r.get("accepted_action") else None) for r in calls]
    parsed_answers = [_parsed_answer(r)[0] for r in calls]
    abstentions = sum(bool(answer is not None and
                           (answer.get("intent") == "unknown" or answer.get("action") == "none"))
                      for answer in parsed_answers)
    parse_failures = sum(r.get("parse_status") not in ("ok", "lenient_ok") for r in calls)
    full_ids = [str(r["id"]) for r in split_rows]
    cohort_set = set(tail_ids)
    unqueried = [r for r in split_rows if str(r["id"]) not in cohort_set]
    deadline_predictions, deadline_counts = _deadline_replay(fallback, run["call_map"])
    return {
        "run_id": run["run_id"], "model_requested": run["model"],
        "model_returned_counts": _count(r.get("model_returned") for r in calls),
        "provider_returned_counts": _count(r.get("provider_returned") for r in calls),
        "attempted": attempts, "accepted_actions": accepted, "changed_actions": changed,
        "model_abstentions": abstentions, "parse_failures_or_invalid_outputs": parse_failures,
        "unaccepted_actions_including_failures_and_mask_rejections": attempts - accepted,
        "masked_rejections": sum(bool(r.get("masked_rejection")) for r in calls),
        "status_counts": _count(statuses), "parse_status_counts": _count(r.get("parse_status") for r in calls),
        "error_type_counts": _count(x for x in errors if x is not None),
        "latency_s": {"p50": float(np.percentile(latencies, 50)) if latencies else None,
                      "p95": float(np.percentile(latencies, 95)) if latencies else None,
                      "max": max(latencies) if latencies else None,
                      "within_production_4s_fraction": float(np.mean(np.asarray(latencies) <= 4.0)) if latencies else None,
                      "timed_calls": len(latencies), "missing_calls": attempts - len(latencies),
                      "measured_interval": "provider call only; hosted deadline 30s; excludes meter refresh"},
        "usage_tokens": token_sums,
        "costs_usd": {"gateway_reported": gateway,
                       "list_uncached": list_cost,
                       "reservations_started_total": float(sum(reservations)),
                       "liability_records_total": float(sum(liabilities)),
                       "run_gateway_usage_delta": run["summary"].get("run_gateway_usage_delta")},
        "selected_tail": _paired_accuracy(tail_rows, fallback, run["predictions"], run["run_id"], "tail"),
        "full_split": _paired_accuracy(split_rows, fallback, run["predictions"], run["run_id"], "full split"),
        "deadline_4s_action_replay": {**deadline_counts,
                                      "selected_tail": _paired_accuracy(tail_rows, fallback, deadline_predictions,
                                                                        run["run_id"], "4s tail"),
                                      "full_split": _paired_accuracy(split_rows, fallback, deadline_predictions,
                                                                     run["run_id"], "4s full split")},
        "unqueried_rows_unchanged": len(unqueried),
        "cohort_sha256": run["summary"]["cohort_sha256"],
    }


def _load_complete_context(cohort_arg: str):
    cohort, digest, cohort_path = _load_cohort(cohort_arg)
    split = cohort.get("split")
    if not isinstance(split, str):
        raise SystemExit("Frozen cohort has no split name")
    rows = list(M.partition(split))
    full_ids = [str(r["id"]) for r in rows]
    items = cohort["items"]
    item_ids = [item.get("id") for item in items]
    if (len(item_ids) != EXPECTED_ATTEMPTS or len(item_ids) != len(set(item_ids))
            or any(not isinstance(i, str) for i in item_ids)):
        raise SystemExit("Frozen cohort item IDs must be unique strings")
    if len(items) != EXPECTED_ATTEMPTS or cohort.get("selected_count") != EXPECTED_ATTEMPTS:
        raise SystemExit(f"Frozen cohort must contain exactly {EXPECTED_ATTEMPTS} sampled rows")
    if set(item_ids) - set(full_ids):
        raise SystemExit("Frozen cohort has IDs outside model_common.partition(split)")
    fallback_path = OUT / f"fallback_{split}.jsonl"
    fallback_rows = _read_jsonl(fallback_path)
    fallback = _check_output_shape(fallback_rows, full_ids, "frozen full-split CPU fallback")
    for item in items:
        reference = item.get("reference")
        if not isinstance(reference, dict) or fallback[item["id"]].get("action") != reference.get("action"):
            raise SystemExit(f"Frozen reference/fallback action mismatch for {item['id']}")
    return cohort, digest, cohort_path, rows, fallback, _unique_rows(items, "cohort items")


def select(args) -> dict:
    if args.runs != list(SELECTION_RUNS):
        raise SystemExit(f"Selection requires these frozen runs in this order: {', '.join(SELECTION_RUNS)}")
    cohort, digest, cohort_path, rows, fallback, items = _load_complete_context(args.cohort)
    if cohort.get("split") != "selection":
        raise SystemExit("Selection choice can only use the explicit selection cohort")
    runs = [_load_run(run_id, cohort, digest, cohort_path, rows, fallback, items) for run_id in args.runs]
    reports = [_run_metrics(run, cohort, rows, fallback) for run in runs]
    chosen, cost_tie_resolved, equal = _choose_selection(reports)
    tie_cost = chosen["costs_usd"]["gateway_reported"]["complete_total"]
    choice = {"selection_split": cohort["split"], "cohort_path": str(cohort_path.relative_to(ROOT)),
              "cohort_sha256": digest, "expected_attempts_per_run": EXPECTED_ATTEMPTS,
              "selection_rule": "highest full-cohort action accuracy; compare gateway cost only when every accuracy-tied candidate has a complete bill; if any is unknown retain selection-luna if tied, else use run_id",
              "selected_run": chosen["run_id"], "selected_model": chosen["model_requested"],
              "selected_cost_gateway_reported_usd": tie_cost,
              "tie_cost_unknown": tie_cost is None, "cost_tie_resolved": cost_tie_resolved,
              "tied_accuracy_runs": [r["run_id"] for r in equal],
              "candidates": reports,
              "interpretation": "action-only selection cohort result; failures retained as frozen CPU fallbacks; review flags/confidence are illustrative and not calibrated for adoption",
              "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out = HOSTED / "selection-choice.json"
    if out.exists():
        raise SystemExit(f"Frozen selection choice already exists; preserve it and choose a new output path: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(choice, indent=2) + "\n", encoding="utf-8")
    return choice


def compare(args) -> dict:
    cohort, digest, cohort_path, rows, fallback, items = _load_complete_context(args.cohort)
    runs = [_load_run(run_id, cohort, digest, cohort_path, rows, fallback, items) for run_id in args.runs]
    row_by_id = {str(row["id"]): row for row in rows}
    tail_rows = [row_by_id[item["id"]] for item in cohort["items"]]
    pairwise = {}
    for i, left in enumerate(runs):
        for right in runs[i + 1:]:
            key = f"{left['run_id']}__vs__{right['run_id']}"
            left_deadline, _ = _deadline_replay(fallback, left["call_map"])
            right_deadline, _ = _deadline_replay(fallback, right["call_map"])
            pairwise[key] = {
                "selected_tail": _paired_accuracy(tail_rows, left["predictions"], right["predictions"],
                                                   right["run_id"], "pairwise tail"),
                "full_split": _paired_accuracy(rows, left["predictions"], right["predictions"],
                                                right["run_id"], "pairwise full split"),
                "deadline_4s_selected_tail": _paired_accuracy(tail_rows, left_deadline, right_deadline,
                                                               right["run_id"], "pairwise 4s tail"),
                "deadline_4s_full_split": _paired_accuracy(rows, left_deadline, right_deadline,
                                                            right["run_id"], "pairwise 4s full split"),
            }
    report = {"split": cohort["split"], "cohort_path": str(cohort_path.relative_to(ROOT)),
              "cohort_sha256": digest, "expected_attempts_per_run": EXPECTED_ATTEMPTS,
              "runs": [_run_metrics(run, cohort, rows, fallback) for run in runs],
              "pairwise_hosted_runs": pairwise,
              "interpretation": "action-only comparison. Failures remain CPU fallbacks; unqueried rows are unchanged. Review/confidence fields are illustrative, not a calibrated production policy.",
              "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out = Path(args.output) if args.output else HOSTED / f"{cohort['split']}-comparison-{digest[:12]}.json"
    out = out if out.is_absolute() else ROOT / out
    out = out.resolve()
    if not out.is_relative_to(ROOT):
        raise SystemExit("Report output path must stay inside the repository")
    if out.exists():
        raise SystemExit(f"Report already exists; refusing to overwrite: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    selection = sub.add_parser("select", help="score complete hosted selection runs and freeze the choice")
    selection.add_argument("--cohort", default="results/model-exploration-20261017/cohorts/selection.json")
    selection.add_argument("--runs", nargs="+", default=list(SELECTION_RUNS))
    selection.set_defaults(func=select)
    comparison = sub.add_parser("compare", help="score explicitly requested complete runs on an explicit cohort")
    comparison.add_argument("--cohort", required=True)
    comparison.add_argument("--runs", nargs="+", required=True)
    comparison.add_argument("--output")
    comparison.set_defaults(func=compare)
    args = parser.parse_args()
    result = args.func(args)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
