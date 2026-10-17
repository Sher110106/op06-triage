"""Bounded hosted, action-only comparison on a frozen low-confidence cohort.

No API work occurs during ``prepare``. Freeze one cohort and its prompts, then run
one exact OpenRouter model ID per run. Every row attempt, including failures, is
recorded once. Failed or invalid answers retain the CPU action. This is an
offline action-quality comparison, not a calibrated pipeline or latency test.

Examples (from solution/):
  python experiments/model_exploration/hosted.py prepare --split selection --limit 300
  python experiments/model_exploration/hosted.py run --key-file /outside/openrouter.key \\
      --model provider/model-snapshot --run-id model-a --cohort results/model-exploration-20261017/cohorts/selection.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from datetime import datetime, timezone
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from triage import gate as G  # noqa: E402
from triage import prompts as P  # noqa: E402
from triage import text as T  # noqa: E402
from triage.policy import load as load_policy  # noqa: E402
from triage.retrieval import Retriever  # noqa: E402
from experiments.model_exploration import model_common as M  # noqa: E402

OUT = M.OUT
RETRIEVAL_PATH = M.MODEL_DIR / "retrieval.joblib"
SEED = 20261023
CPU_GATE = 0.642841
GLOBAL_BUDGET_USD = 5.0
MAX_OUTPUT_TOKENS = 256
OVERHEAD_BYTES = 1024
MAX_RESPONSE_BYTES = 256 * 1024
MAX_USAGE_TOKENS = 10_000_000
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def _sha(text: str | bytes) -> str:
    if isinstance(text, str):
        text = text.encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def _reference_path(split: str) -> Path:
    return OUT / f"reference_{split}.jsonl"


def _load_reference(split: str, rows: list[dict]) -> list[dict]:
    path = _reference_path(split)
    if not path.exists():
        raise SystemExit(f"Missing precomputed reference predictions: {path}")
    predictions = _jsonl(path)
    row_ids = [r["id"] for r in rows]
    pred_ids = [r.get("id") for r in predictions]
    if pred_ids != row_ids:
        raise SystemExit(f"Reference IDs/order do not exactly match model_common.partition({split!r})")
    for p in predictions:
        if not isinstance(p.get("confidence"), (int, float)) or not math.isfinite(float(p["confidence"])):
            raise SystemExit(f"Reference row {p['id']} has no finite confidence")
        if not isinstance(p.get("action"), str) or not isinstance(p.get("intent"), str):
            raise SystemExit(f"Reference row {p['id']} lacks intent/action")
    return predictions


def _load_gate_details(split: str, rows: list[dict]) -> dict[str, dict]:
    path = OUT / "reference" / f"details_{split}.json"
    if not path.exists():
        raise SystemExit(f"Missing precomputed reference gate details: {path}")
    details = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(details, dict):
        raise SystemExit(f"Reference gate details must map row IDs to details: {path}")
    expected = {r["id"] for r in rows}
    if set(details) != expected:
        raise SystemExit(f"Reference gate-detail IDs do not exactly match model_common.partition({split!r})")
    for rid, item in details.items():
        if not isinstance(item, dict):
            raise SystemExit(f"Gate details for {rid} must be an object")
        if not isinstance(item.get("abstained"), bool):
            raise SystemExit(f"abstained for {rid} must be boolean")
        confidence_raw = item.get("gate_confidence")
        if confidence_raw is None and item["abstained"]:
            confidence = None
        else:
            try:
                confidence = _finite_nonnegative(confidence_raw, f"{rid}.gate_confidence")
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            if confidence is None or confidence > 1:
                raise SystemExit(f"Gate confidence for {rid} must be in [0, 1]")
        flags = item.get("gate_flags")
        if not isinstance(flags, list) or any(not isinstance(flag, str) for flag in flags):
            raise SystemExit(f"Gate flags for {rid} must be a list of strings")
    return details


def _fit_retriever() -> Retriever:
    rows = M.partition("fit")
    fit_rows = []
    for row in rows:
        model_row = M.model_input(row)
        model_row.update(intent=row["intent"], action=row["action"])
        fit_rows.append(model_row)
    if RETRIEVAL_PATH.exists():
        cached = Retriever.load(RETRIEVAL_PATH)
        expected = [r["id"] for r in fit_rows]
        if cached.ids == expected:
            return cached
    retriever = Retriever.build(fit_rows)
    retriever.meta.update({"source": "model_common.partition('fit')", "fit_ids_sha256": _sha("\n".join(r["id"] for r in fit_rows))})
    retriever.save(RETRIEVAL_PATH)
    return retriever


def _prepare(args) -> None:
    path = OUT / "cohorts" / f"{args.split}.json"
    fallback = OUT / f"fallback_{args.split}.jsonl"
    if path.exists():
        raise SystemExit(f"Frozen cohort already exists: {path}; preserve it and use a new cohort name")
    if fallback.exists():
        raise SystemExit(f"Fallback already exists: {fallback}; refusing to overwrite it")
    split_rows = M.partition(args.split)
    refs = _load_reference(args.split, split_rows)
    gate_details = _load_gate_details(args.split, split_rows)
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    eligible = []
    for row, pred in zip(split_rows, refs):
        detail = gate_details[row["id"]]
        flags = set(detail["gate_flags"])
        if detail["abstained"] or detail["gate_confidence"] is None or detail["gate_confidence"] >= CPU_GATE or flags & G.ROUTE_FLAGS:
            continue
        eligible.append((row, pred, detail))
    rng = random.Random(SEED)
    rng.shuffle(eligible)
    selected = eligible[:args.limit]
    retriever = _fit_retriever()
    policy = load_policy()
    if policy.version != "2026-10-17.2":
        raise SystemExit(f"Expected policy 2026-10-17.2, found {policy.version}")
    static = P.static_prompt()
    schema = P.output_schema()
    items = []
    for row, pred, detail in selected:
        visible = M.model_input(row)
        examples = retriever.examples(visible, k=8)
        masked = sorted(policy.masked_actions(T.observed_actions(visible)))
        nonce = "cohort" + _sha(row["id"])[:16]
        dynamic, returned_nonce = P.dynamic_prompt(visible, examples, nonce=nonce)
        assert returned_nonce == nonce
        items.append({"id": row["id"], "reference": pred, "dynamic_prompt": dynamic,
                      "nonce": nonce, "retrieval_example_ids": [e["id"] for e in examples],
                      "gate_confidence": detail["gate_confidence"], "gate_flags": detail["gate_flags"],
                      "abstained": detail["abstained"],
                      "policy_version": policy.version, "masked_actions": masked})
    payload = {
        "format": 1, "purpose": "frozen hosted action-only comparison; not a calibrated pipeline or production latency test",
        "split": args.split, "seed": SEED, "limit_requested": args.limit,
        "eligible_count": len(eligible), "selected_count": len(items),
        "confidence_gate": CPU_GATE, "selection_rule": "deterministic random sample of uncapped gate_confidence < gate, no ROUTE flag, and not abstained",
        "policy_version": policy.version,
        "reference_path": str(_reference_path(args.split).relative_to(ROOT)),
        "reference_sha256": _sha(_reference_path(args.split).read_bytes()),
        "retrieval_path": str(RETRIEVAL_PATH.relative_to(ROOT)),
        "retrieval_meta": retriever.meta,
        "static_prompt": static, "static_prompt_sha256": _sha(static), "output_schema": schema,
        "items": items,
    }
    M.write_json(path, payload)
    # Full-split CPU fallback is saved separately and never altered by the hosted run.
    with fallback.open("w", encoding="utf-8") as f:
        for p in refs:
            f.write(json.dumps(_response(p), ensure_ascii=False) + "\n")
    print(json.dumps({"cohort": str(path.relative_to(ROOT)), "fallback": str(fallback.relative_to(ROOT)),
                      "eligible": len(eligible), "selected": len(items), "retrieval_fit_rows": len(retriever.ids),
                      "cohort_sha256": _sha(path.read_bytes())}, indent=2))


def _response(pred: dict) -> dict:
    return {"id": pred["id"], "intent": pred["intent"], "action": pred["action"],
            "confidence": pred["confidence"], "needs_human": bool(pred.get("needs_human", False))}


def _read_models() -> list[dict]:
    path = OUT / "api_models.json"
    if not path.exists():
        raise SystemExit(f"Missing saved model price snapshot: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("api_models.json must contain the saved model list")
    return data


def _finite_nonnegative(value: Any, name: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a numeric JSON value")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _positive_rate(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} must be a positive numeric price")
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a positive numeric price") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _prices(model_id: str) -> dict:
    matches = [m for m in _read_models() if m.get("id") == model_id]
    if len(matches) != 1:
        raise SystemExit(f"Model ID must exactly match one saved api_models.json entry: {model_id}")
    pricing = matches[0].get("pricing")
    if not isinstance(pricing, dict):
        raise SystemExit(f"No saved prices for exact model ID {model_id}")
    threshold_tiers, time_tiers, all_rates = [], [], []
    try:
        base = {"min_prompt_tokens": 0,
                "prompt": _positive_rate(pricing.get("prompt"), "base prompt rate"),
                "completion": _positive_rate(pricing.get("completion"), "base completion rate")}
        threshold_tiers.append(base); all_rates.append(base)
        for override in pricing.get("overrides", []):
            if not isinstance(override, dict):
                raise ValueError("price override must be an object")
            rate = {"prompt": _positive_rate(override.get("prompt"), "override prompt rate"),
                    "completion": _positive_rate(override.get("completion"), "override completion rate")}
            all_rates.append(rate)
            if "min_prompt_tokens" in override:
                raw_min = override["min_prompt_tokens"]
                if isinstance(raw_min, bool) or not isinstance(raw_min, int) or raw_min < 0:
                    raise ValueError("min_prompt_tokens must be a nonnegative integer")
                threshold_tiers.append(dict(rate, min_prompt_tokens=raw_min))
            elif "utc_start" in override or "utc_end" in override or "utc_days" in override:
                start = override.get("utc_start", 0)
                end = override.get("utc_end", 0)
                if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 2400 for v in (start, end)):
                    raise ValueError("UTC price-window bounds must be integers from 0 to 2400")
                days = override.get("utc_days")
                if days is not None and (not isinstance(days, list) or any(d not in
                        ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday") for d in days)):
                    raise ValueError("utc_days must contain valid weekday names")
                time_tiers.append(dict(rate, utc_start=start, utc_end=end, utc_days=days))
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"Invalid or unpriced saved rates for {model_id}: {exc}") from exc
    threshold_tiers.sort(key=lambda x: x["min_prompt_tokens"])
    endpoint_file = OUT / "api_endpoints.json"
    if not endpoint_file.exists():
        raise SystemExit(f"Missing saved provider endpoint prices: {endpoint_file}")
    endpoints_by_model = json.loads(endpoint_file.read_text(encoding="utf-8"))
    endpoint_entry = endpoints_by_model.get(model_id) if isinstance(endpoints_by_model, dict) else None
    endpoints = endpoint_entry.get("endpoints") if isinstance(endpoint_entry, dict) else None
    if not isinstance(endpoints, list) or not endpoints:
        raise SystemExit(f"No saved provider endpoints for exact model ID {model_id}")
    endpoint_rates = []
    try:
        for endpoint in endpoints:
            endpoint_pricing = endpoint.get("pricing") if isinstance(endpoint, dict) else None
            if not isinstance(endpoint_pricing, dict):
                continue
            prompt_rate = _positive_rate(endpoint_pricing.get("prompt"), "endpoint prompt rate")
            completion_rate = _positive_rate(endpoint_pricing.get("completion"), "endpoint completion rate")
            endpoint_rates.append({"provider_name": endpoint.get("provider_name"),
                                   "prompt": prompt_rate, "completion": completion_rate})
            for override in endpoint_pricing.get("overrides", []):
                if not isinstance(override, dict):
                    raise ValueError("endpoint price override must be an object")
                endpoint_rates.append({"provider_name": endpoint.get("provider_name"),
                                       "prompt": _positive_rate(override.get("prompt"), "endpoint override prompt rate"),
                                       "completion": _positive_rate(override.get("completion"), "endpoint override completion rate")})
    except ValueError as exc:
        raise SystemExit(f"Invalid saved provider endpoint price for {model_id}: {exc}") from exc
    if not endpoint_rates:
        raise SystemExit(f"No positive saved provider endpoint prices for exact model ID {model_id}")
    endpoint_prompt_max = max(t["prompt"] for t in endpoint_rates)
    endpoint_completion_max = max(t["completion"] for t in endpoint_rates)
    return {"threshold_tiers": threshold_tiers, "time_tiers": time_tiers,
            # Maximum across all saved applicable prices is used only for the pre-call bound.
            "max_prompt_rate": max(t["prompt"] for t in all_rates),
            "max_completion_rate": max(t["completion"] for t in all_rates),
            "endpoint_prompt_max": endpoint_prompt_max,
            "endpoint_completion_max": endpoint_completion_max,
            "endpoint_count": len(endpoints),
            "priced_endpoint_tier_count": len(endpoint_rates),
            "reservation_prompt_rate": max(max(t["prompt"] for t in all_rates), endpoint_prompt_max),
            "reservation_completion_rate": max(max(t["completion"] for t in all_rates), endpoint_completion_max)}


def _rates_for_prompt(prices: dict, prompt_tokens: int, at_utc: datetime | None = None) -> tuple[float, float]:
    tier = prices["threshold_tiers"][0]
    for candidate in prices["threshold_tiers"][1:]:
        if prompt_tokens < candidate["min_prompt_tokens"]:
            break
        tier = candidate
    at_utc = at_utc or datetime.now(timezone.utc)
    minute_hour = at_utc.hour * 100 + at_utc.minute
    weekday = at_utc.strftime("%A").lower()
    for candidate in prices["time_tiers"]:
        start, end = candidate["utc_start"], candidate["utc_end"]
        in_time = (start <= minute_hour < end) if start < end else (minute_hour >= start or minute_hour < end)
        if in_time and (candidate["utc_days"] is None or weekday in candidate["utc_days"]):
            tier = candidate
            break
    return tier["prompt"], tier["completion"]


def _valid_token_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > MAX_USAGE_TOKENS:
        return None
    return value


def _response_liability(usage: dict | None, prices: dict, at_utc: datetime | None = None) -> tuple[float | None, dict, list[str]]:
    """Return uncached list estimate, sanitized token/cost values, and invalid field names."""
    usage = usage if isinstance(usage, dict) else {}
    cleaned, invalid = {}, []
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(name)
        parsed = _valid_token_count(value)
        cleaned[name] = parsed
        if value is not None and parsed is None:
            invalid.append(name)
    raw_cost = usage.get("cost")
    try:
        cost = _finite_nonnegative(raw_cost, "usage.cost", allow_none=True)
    except ValueError:
        cost = None
        invalid.append("cost")
    cleaned["cost"] = cost
    prompt_tokens, completion_tokens = cleaned["prompt_tokens"], cleaned["completion_tokens"]
    estimate = None
    if prompt_tokens is not None and completion_tokens is not None:
        in_rate, out_rate = _rates_for_prompt(prices, prompt_tokens, at_utc)
        # Saved OpenRouter `pricing` rates are USD per token, unlike the legacy probe's per-million table.
        estimate = prompt_tokens * in_rate + completion_tokens * out_rate
        if not math.isfinite(estimate) or estimate < 0:
            estimate = None
            invalid.append("list_estimate")
    return estimate, cleaned, invalid


def _liability(record: dict, prices: dict | None = None) -> float:
    """Use max(valid gateway usage.cost, applicable-tier estimate), or reserve if usage is unknown."""
    reserve = _finite_nonnegative(record.get("reserved_usd"), "reserved_usd")
    stored = record.get("liability_usd")
    if stored is not None:
        return _finite_nonnegative(stored, "liability_usd")
    if record.get("status") != "ok":
        return reserve
    usage = record.get("usage")
    if prices is not None and isinstance(usage, dict):
        try:
            stamp = record.get("request_started_utc")
            at_utc = datetime.fromisoformat(stamp.replace("Z", "+00:00")) if stamp else None
            estimate, clean, _ = _response_liability(usage, prices, at_utc)
            cost = clean.get("cost")
            if estimate is not None or cost is not None:
                return max(estimate or 0.0, cost or 0.0)
        except (ValueError, TypeError):
            pass
    return reserve


def _meter(key: str) -> dict:
    req = urllib.request.Request("https://openrouter.ai/api/v1/auth/key",
                                 headers={"Authorization": "Bearer " + key, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    d = data.get("data", {})
    try:
        remaining = _finite_nonnegative(d.get("limit_remaining"), "gateway.limit_remaining")
        limit = _finite_nonnegative(d.get("limit"), "gateway.limit", allow_none=True)
        usage = _finite_nonnegative(d.get("usage"), "gateway.usage", allow_none=True)
    except ValueError as exc:
        raise RuntimeError("gateway_meter_invalid_numeric_value") from exc
    if remaining is None:
        raise RuntimeError("gateway_meter_missing_limit_remaining")
    return {"limit": limit, "limit_remaining": remaining, "usage": usage}


def _phase_spend(before: dict, now: dict) -> float:
    before_remaining = _finite_nonnegative(before.get("limit_remaining"), "baseline.limit_remaining")
    now_remaining = _finite_nonnegative(now.get("limit_remaining"), "gateway.limit_remaining")
    if before_remaining is None or now_remaining is None:
        raise ValueError("missing gateway remaining balance")
    deltas = [before_remaining - now_remaining]
    before_usage = _finite_nonnegative(before.get("usage"), "baseline.usage", allow_none=True)
    now_usage = _finite_nonnegative(now.get("usage"), "gateway.usage", allow_none=True)
    if before_usage is not None and now_usage is not None:
        deltas.append(now_usage - before_usage)
    return max(0.0, *deltas)


def _phase_liabilities(prices_cache: dict[str, dict] | None = None) -> float:
    """Sum completed-call liabilities and reservations for attempts with unknown usage."""
    total = 0.0
    price_cache = prices_cache if prices_cache is not None else {}
    for run_dir in (OUT / "hosted").glob("*"):
        if not run_dir.is_dir():
            continue
        raw_path = run_dir / "raw_calls.jsonl"
        started_path = run_dir / "started_calls.jsonl"
        finals = _jsonl(raw_path) if raw_path.exists() else []
        final_ids = {r.get("id") for r in finals}
        for rec in finals:
            if rec.get("liability_usd") is not None:
                total += _finite_nonnegative(rec["liability_usd"], "liability_usd")
                continue
            model_id = rec.get("model_requested") or rec.get("model")
            rec_prices = None
            if model_id:
                try:
                    if model_id not in price_cache:
                        price_cache[model_id] = _prices(model_id)
                    rec_prices = price_cache[model_id]
                except SystemExit:
                    pass
            total += _liability(rec, rec_prices)
        started = _jsonl(started_path) if started_path.exists() else []
        for rec in started:
            if rec.get("id") not in final_ids:
                total += _finite_nonnegative(rec.get("reserved_usd"), "started.reserved_usd")
    if not math.isfinite(total) or total < 0:
        raise ValueError("phase liabilities must be finite and nonnegative")
    return total


def _reservation(serialized_request: str, prices: dict) -> float:
    # Bound the complete UTF-8 JSON request (including schema and message wrappers), then add protocol overhead.
    in_tokens_bound = len(serialized_request.encode("utf-8")) + OVERHEAD_BYTES
    max_output = json.loads(serialized_request).get("max_tokens", MAX_OUTPUT_TOKENS)
    if isinstance(max_output, bool) or not isinstance(max_output, int) or not 1 <= max_output <= 4096:
        raise ValueError("invalid maximum output token bound")
    result = in_tokens_bound * prices["reservation_prompt_rate"] + max_output * prices["reservation_completion_rate"]
    if not math.isfinite(result) or result < 0:
        raise ValueError("reservation must be finite and nonnegative")
    return result


def request_profile(model_id: str) -> dict:
    # Gemini 3.8 requires thinking; low is its lowest supported level. Its
    # provider-native sampling defaults replace the deprecated temperature knob.
    if model_id == "google/gemini-3.8-flash":
        return {"max_tokens": 1024, "reasoning": {"effort": "low"}}
    return {"max_tokens": MAX_OUTPUT_TOKENS, "reasoning": {"effort": "none"}, "temperature": 0}


def _cohort(path: Path) -> tuple[dict, str]:
    path = path.resolve()
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read cohort JSON: {type(exc).__name__}")
    if payload.get("format") != 1 or not isinstance(payload.get("items"), list):
        raise SystemExit("Unsupported or malformed frozen cohort")
    if _sha(payload["static_prompt"]) != payload.get("static_prompt_sha256"):
        raise SystemExit("Frozen cohort static prompt hash mismatch")
    return payload, _sha(raw)


def _run(args) -> None:
    if not SAFE_RUN_ID.fullmatch(args.run_id):
        raise SystemExit("--run-id must use 1-64 letters, digits, hyphens or underscores")
    if isinstance(args.timeout, bool) or not isinstance(args.timeout, int) or args.timeout <= 0 or args.timeout > 120:
        raise SystemExit("--timeout must be an integer in (0, 120]")
    try:
        run_cap = _finite_nonnegative(args.max_spend_usd, "--max-spend-usd")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if run_cap is None or run_cap <= 0:
        raise SystemExit("--max-spend-usd must be positive")
    if run_cap > GLOBAL_BUDGET_USD:
        raise SystemExit(f"--max-spend-usd cannot exceed the shared ${GLOBAL_BUDGET_USD:.0f} phase cap")
    key_path = Path(args.key_file).resolve()
    if key_path.is_relative_to(ROOT):
        raise SystemExit("Credential file must be outside the repository")
    key = key_path.read_text(encoding="utf-8").strip()
    if not key:
        raise SystemExit("Credential file is empty")
    model_id = args.model.strip()
    if not model_id or " " in model_id:
        raise SystemExit("--model must be one exact saved model ID")
    prices = _prices(model_id)
    cohort_path = Path(args.cohort)
    if not cohort_path.is_absolute():
        cohort_path = ROOT / cohort_path
    cohort, cohort_hash = _cohort(cohort_path)
    fallback_path = OUT / f"fallback_{cohort['split']}.jsonl"
    if not fallback_path.exists():
        raise SystemExit(f"Missing frozen CPU fallback: {fallback_path}")
    frozen_fallback = _jsonl(fallback_path)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        cohort["items"] = cohort["items"][:args.limit]
    items = cohort["items"]
    if not items:
        raise SystemExit("Cohort is empty")
    run_dir = OUT / "hosted" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    calls_path = run_dir / "raw_calls.jsonl"
    started_path = run_dir / "started_calls.jsonl"
    meta_path = run_dir / "run.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("model") != model_id or old.get("cohort_sha256") != cohort_hash:
            raise SystemExit("run-id already belongs to a different model or frozen cohort")
    else:
        M.write_json(meta_path, {"format": 1, "model": model_id, "cohort": str(cohort_path),
                                 "cohort_sha256": cohort_hash, "split": cohort["split"],
                                 "purpose": "offline action-only cohort comparison; no calibrated pipeline or production latency claim",
                                 "price_snapshot": str((OUT / "api_models.json").relative_to(ROOT)),
                                 "max_spend_usd": args.max_spend_usd,
                                 "max_tokens": request_profile(model_id)["max_tokens"],
                                 "request_profile": request_profile(model_id),
                                 "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    records = _jsonl(calls_path) if calls_path.exists() else []
    profile = request_profile(model_id)
    for record in records:
        previous = record.get("request") or {}
        if previous and any(previous.get(k) != profile.get(k) for k in ("max_tokens", "reasoning", "temperature")):
            raise SystemExit("Request settings changed; use a new run ID and preserve the previous experiment")
    started_records = _jsonl(started_path) if started_path.exists() else []
    attempted = {r["id"] for r in records} | {r["id"] for r in started_records}
    duplicate_ids = len(records) - len({r["id"] for r in records})
    duplicate_ids += len(started_records) - len({r["id"] for r in started_records})
    if duplicate_ids:
        raise SystemExit("raw_calls.jsonl already contains duplicate row attempts; refusing to resume")
    pending = [x for x in items if x["id"] not in attempted]
    budget_before_file = OUT / "budget_before.json"
    if not budget_before_file.exists():
        raise SystemExit(f"Missing global phase budget baseline: {budget_before_file}")
    baseline = json.loads(budget_before_file.read_text(encoding="utf-8"))
    if "limit_remaining" not in baseline:
        raise SystemExit("budget_before.json lacks limit_remaining")
    try:
        phase_before = {"limit": _finite_nonnegative(baseline.get("limit"), "baseline.limit", allow_none=True),
                        "limit_remaining": _finite_nonnegative(baseline.get("limit_remaining"), "baseline.limit_remaining"),
                        "usage": _finite_nonnegative(baseline.get("usage"), "baseline.usage", allow_none=True)}
    except ValueError as exc:
        raise SystemExit(f"Invalid global phase baseline: {exc}") from exc
    if phase_before["limit_remaining"] is None:
        raise SystemExit("budget_before.json has no valid limit_remaining")
    meter_start = _meter(key)
    phase_start = _phase_spend(phase_before, meter_start)
    if phase_start >= GLOBAL_BUDGET_USD:
        raise SystemExit("Global model-exploration phase budget is already exhausted")
    final_ids = {r["id"] for r in records}
    run_liabilities = sum(_liability(r, prices) for r in records)
    run_liabilities += sum(_finite_nonnegative(r.get("reserved_usd"), "started.reserved_usd")
                           for r in started_records if r["id"] not in final_ids)
    run_reserved_total = sum(_finite_nonnegative(r.get("reserved_usd"), "started.reserved_usd")
                             for r in started_records)
    executed_now = 0
    # Store a pre-call meter snapshot even when resuming, without writing credentials.
    start_meta = {"meter_before": meter_start, "phase_spend_before": phase_start,
                  "already_attempted": len(attempted), "pending": len(pending),
                  "max_tokens": profile["max_tokens"], "request_profile": profile,
                  "max_spend_usd": run_cap}
    meta = json.loads(meta_path.read_text(encoding="utf-8")); meta.update(start_meta)
    M.write_json(meta_path, meta)
    stop_reason = None
    for item in pending:
        rid = item["id"]
        static, dynamic = cohort["static_prompt"], item["dynamic_prompt"]
        if item.get("policy_version") != cohort.get("policy_version") or item.get("policy_version") != "2026-10-17.2":
            raise SystemExit(f"Frozen policy version mismatch for {rid}")
        masked_actions = item.get("masked_actions")
        if not isinstance(masked_actions, list) or any(not isinstance(a, str) for a in masked_actions):
            raise SystemExit(f"Frozen mask is malformed for {rid}")
        body = {
            "model": model_id, "messages": [{"role": "system", "content": static},
                                               {"role": "user", "content": dynamic}],
            **request_profile(model_id),
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "op06_triage", "strict": True, "schema": cohort["output_schema"]}},
            "usage": {"include": True},
        }
        request_json = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
        reserve = _reservation(request_json, prices)
        meter = _meter(key)
        phase_spent = _phase_spend(phase_before, meter)
        if run_liabilities + reserve > run_cap:
            stop_reason = "per_run_reservation_cap"
            break
        phase_liability = _phase_liabilities()
        if max(phase_spent, phase_liability) + reserve > GLOBAL_BUDGET_USD:
            stop_reason = "global_phase_reservation_cap"
            break
        # Metered budget can be lower than its advertised total limit. Do not approach zero.
        if meter["limit_remaining"] <= reserve:
            stop_reason = "gateway_remaining_below_reservation"
            break
        # Persist the reservation before sending the request. If the process dies mid-call,
        # resume will not duplicate the call and the unresolved reservation remains charged.
        _append_jsonl(started_path, {"id": rid, "model_requested": model_id,
                                     "reserved_usd": reserve, "request": body,
                                     "policy_version": item["policy_version"],
                                     "masked_actions": masked_actions})
        started_records.append({"id": rid, "reserved_usd": reserve})
        run_reserved_total += reserve
        request_started_utc = datetime.now(timezone.utc)
        started = time.perf_counter()
        http_status = None; response_body = None; error_class = None
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                                         data=request_json.encode("utf-8"),
                                         headers={"Authorization": "Bearer " + key,
                                                  "Content-Type": "application/json", "Accept": "application/json"},
                                         method="POST")
            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                http_status = response.status
                response_bytes = response.read(MAX_RESPONSE_BYTES + 1)
                if len(response_bytes) > MAX_RESPONSE_BYTES:
                    error_class = "response_too_large"
                else:
                    response_body = json.loads(response_bytes)
        except urllib.error.HTTPError as exc:
            http_status = exc.code
            error_class = "http_error"
            try:
                exc.close()
            except Exception:
                pass
        except urllib.error.URLError as exc:
            error_class = "timeout" if isinstance(exc.reason, TimeoutError) else "connection_error"
        except TimeoutError:
            error_class = "timeout"
        except (json.JSONDecodeError, UnicodeDecodeError):
            error_class = "invalid_gateway_json"
        except Exception as exc:
            error_class = type(exc).__name__
        latency = time.perf_counter() - started
        text = None; usage = None; model_returned = None; provider_returned = None; request_id = None
        if isinstance(response_body, dict):
            model_returned = response_body.get("model")
            request_id = response_body.get("id")
            choices = response_body.get("choices") or []
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                message = choices[0].get("message")
                text = message.get("content") if isinstance(message, dict) else None
                provider_returned = choices[0].get("provider") or response_body.get("provider")
            raw_usage = response_body.get("usage")
            if isinstance(raw_usage, dict):
                usage = {k: raw_usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")}
        if http_status == 200 and error_class is None and isinstance(text, str):
            try:
                parsed, parse_status = P.parse_output(text)
            except Exception:
                parsed, parse_status = None, "invalid_response_shape"
        else:
            parsed, parse_status = None, "call_failed" if error_class else "invalid_response_shape"
        parse_ok = parsed is not None and parse_status in ("ok", "lenient_ok")
        mask_rejected = bool(parse_ok and parsed.get("intent") != "unknown" and parsed.get("action") != "none"
                             and parsed.get("action") in masked_actions)
        accepted = bool(parse_ok and parsed.get("intent") != "unknown" and parsed.get("action") != "none"
                        and parsed.get("action") in P.ACTIONS and not mask_rejected)
        chosen = parsed["action"] if accepted else item["reference"]["action"]
        changed = accepted and chosen != item["reference"]["action"]
        record = {
            "id": rid, "model_requested": model_id, "model_returned": model_returned,
            "provider_returned": provider_returned, "request_id": request_id,
            "status": ("ok" if accepted else (error_class or "invalid_or_abstained"))
                      if http_status == 200 else (error_class or "http_error"),
            "http_status": http_status, "error_class": error_class,
            "latency_s": latency, "attempt": 1, "reserved_usd": reserve,
            "request_started_utc": request_started_utc.isoformat().replace("+00:00", "Z"),
            "usage": usage, "cost_list_uncached_usd": None,
            "gateway_reported_cost_usd": None, "liability_usd": reserve,
            "liability_source": "reservation_unknown_usage", "usage_invalid_fields": [],
            "parse_status": parse_status, "accepted_action": bool(accepted),
            "policy_version": item["policy_version"], "masked_actions": masked_actions,
            "masked_rejection": mask_rejected,
            "action_changed": bool(changed), "action_selected": chosen,
            "reference_action": item["reference"]["action"],
            "retrieval_example_ids": item["retrieval_example_ids"],
            "prompt_sha256": _sha(static + "\0" + dynamic),
            "request": body,
            # Save only successful structured response content. HTTP error bodies can echo credentials or prompts.
            "response": response_body if http_status == 200 else None,
        }
        if usage is not None:
            estimate, sanitized_usage, invalid_fields = _response_liability(usage, prices, request_started_utc)
            record["usage"] = sanitized_usage
            record["usage_invalid_fields"] = invalid_fields
            if estimate is not None:
                record["cost_list_uncached_usd"] = estimate
            reported_cost = sanitized_usage.get("cost")
            if reported_cost is not None:
                record["gateway_reported_cost_usd"] = reported_cost
            if estimate is not None or reported_cost is not None:
                record["liability_usd"] = max(estimate or 0.0, reported_cost or 0.0)
                record["liability_source"] = "max_gateway_cost_and_uncached_list_estimate"
            if invalid_fields:
                accepted = False
                record.update(status="invalid_usage", accepted_action=False, action_changed=False,
                              action_selected=item["reference"]["action"], liability_usd=reserve,
                              liability_source="reservation_invalid_usage")
        run_liabilities += _finite_nonnegative(record["liability_usd"], "liability_usd")
        executed_now += 1
        _append_jsonl(calls_path, record)
        records.append(record)
        # Refresh gateway meter after every attempt. Failing to read it halts further calls.
        try:
            meter_after = _meter(key)
            phase_spent = _phase_spend(phase_before, meter_after)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["meter_after_last_attempt"] = meter_after
            meta["phase_spend_after_last_attempt"] = phase_spent
            meta["run_liabilities_usd"] = run_liabilities
            meta["run_total_reservations_descriptive_usd"] = run_reserved_total
            M.write_json(meta_path, meta)
            if phase_spent >= GLOBAL_BUDGET_USD:
                stop_reason = "global_phase_budget_reached_after_call"
                break
        except Exception as exc:
            stop_reason = f"meter_after_failed:{type(exc).__name__}"
            break
        if http_status != 200 or not accepted:
            # Preserve this attempt as a failure/abstention and proceed to the next distinct row.
            continue
    # Rebuild, rather than append, the full-split diagnostic file so resume stays idempotent.
    out = run_dir / f"predictions_{cohort['split']}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        calls = {r["id"]: r for r in records}
        for reference in frozen_fallback:
            pred = _response(reference)
            call = calls.get(reference["id"])
            if call and call.get("accepted_action") and call.get("action_changed"):
                pred["action"] = call["action_selected"]
                pred["needs_human"] = True
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    meter_end = _meter(key)
    final_records = _jsonl(calls_path) if calls_path.exists() else []
    summary = {"purpose": "offline action-only cohort comparison; not a calibrated pipeline or production latency test",
               "run_id": args.run_id, "model": model_id, "cohort_sha256": cohort_hash,
               "selected_rows": len(items), "attempted_rows": len(attempted | {r["id"] for r in final_records}),
               "new_attempts": executed_now,
               "accepted_actions": sum(bool(r.get("accepted_action")) for r in final_records),
               "changed_actions": sum(bool(r.get("action_changed")) for r in final_records),
               "run_liabilities_usd": run_liabilities,
               "run_total_reservations_descriptive_usd": run_reserved_total,
               "meter_start": meter_start, "meter_end": meter_end,
               "run_gateway_usage_delta": _phase_spend(meter_start, meter_end),
               "phase_gateway_delta": _phase_spend(phase_before, meter_end),
               "phase_recorded_liabilities_usd": _phase_liabilities(), "stop_reason": stop_reason,
               "predictions": str(out.relative_to(ROOT)), "raw_calls": str(calls_path.relative_to(ROOT))}
    M.write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="freeze the eligible cohort, prompts, retrieval and reference fallbacks; no API calls")
    prep.add_argument("--split", required=True, choices=("selection", "calibration", "confirmation", "dev", "noisy"))
    prep.add_argument("--limit", type=int, default=300)
    prep.set_defaults(func=_prepare)
    run = sub.add_parser("run", help="run one exact hosted model ID on a frozen cohort")
    run.add_argument("--key-file", required=True, help="credential file outside the repository")
    run.add_argument("--model", required=True, help="exact ID from saved api_models.json")
    run.add_argument("--run-id", required=True)
    run.add_argument("--cohort", required=True)
    run.add_argument("--max-spend-usd", type=float, default=1.5)
    run.add_argument("--limit", type=int, help="bounded smoke subset of the frozen cohort")
    run.add_argument("--timeout", type=int, default=30)
    run.add_argument("--concurrency", type=int, default=1, choices=(1,), help="sequential calls protect the shared phase budget")
    run.set_defaults(func=_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
