"""Load test for the running service: sends rows to POST /triage at a fixed concurrency and times each request.

  .venv/bin/python scripts/loadtest.py [--url http://127.0.0.1:8000] [--concurrency 8] [--inputs rows.jsonl]
      [--limit N] [--repeat 1] [--warmup 20] [--out results/loadtest-latest.json] [--pred-out preds.jsonl]
      [--label NAME] [--timeout-s 30]

Default inputs: dev + dev_noisy + hostile, projected with triage.data.project (id, turn_index, context only).
Latency is measured by the client around the whole request. LLM calls, tokens and cost are the change in GET /metrics
over the run: llm_cost_usd at list price with cache discounts, llm_cost_uncached_usd at list price with none (how
MODELS.md prices a benchmark); the uncached cost and token counts are null when the service does not report them
(an older service). Warm-up requests use a changed id, so they do not fill the service's idempotency cache for the
measured rows, and a request id starting "lt-warmup-", so the service log tells them apart. A repeated body is
answered from that cache, so --repeat 2 measures cache replays.
Exits 1 when any request fails, returns a status other than 200, or returns a body that is not schema-valid.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage.data import DATA, ROOT, load_jsonl, load_split, project  # noqa: E402


def _grader():
    spec = importlib.util.spec_from_file_location("op06_grader", DATA / "grader.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _grader()


def _project(row):
    ctx = row.get("context") if isinstance(row, dict) else None
    return project(row) if isinstance(ctx, list) and all(isinstance(t, dict) for t in ctx) else row


def load_inputs(path: str | None) -> list:
    rows = load_jsonl(path) if path else load_split("dev") + load_split("noisy") + load_split("hostile")
    return [_project(r) for r in rows]


def _warm_row(row, i: int):
    """A changed id, so the warm-up does not put the measured rows in the service's cache."""
    if isinstance(row, dict) and isinstance(row.get("id"), str):
        return {**row, "id": f"warmup-{i}-{row['id']}"[:256]}
    return row


def _encode(row) -> bytes:
    return json.dumps(row).encode("ascii")                  # ASCII escapes keep lone surrogates sendable


async def _get_json(client: httpx.AsyncClient, path: str) -> dict:
    try:
        r = await client.get(path)
        return {"http_status": r.status_code, **r.json()}
    except Exception as e:                                   # noqa: BLE001 - the report says what went wrong
        return {"error": f"{type(e).__name__}: {e}"}


async def wait_ready(client: httpx.AsyncClient, wait_s: float) -> dict:
    deadline = time.monotonic() + wait_s
    while True:
        h = await _get_json(client, "/healthz")
        if h.get("http_status") == 200 or time.monotonic() > deadline:
            return h
        await asyncio.sleep(0.5)


async def one(client: httpx.AsyncClient, seq: int, row, t0: float, timeout_s: float) -> dict:
    rid = f"lt-{'warmup-' if seq < 0 else ''}{uuid.uuid4().hex}"
    rec = {"seq": seq, "request_id": rid, "id": row.get("id") if isinstance(row, dict) else None}
    body = _encode(row)
    start = time.perf_counter()
    rec["start_s"] = round(start - t0, 6)
    status, pred, err, replay = None, None, None, False
    try:
        async with asyncio.timeout(timeout_s):
            r = await client.post("/triage", content=body,
                                  headers={"Content-Type": "application/json", "X-Request-ID": rid})
            status, text = r.status_code, r.text
        replay = r.headers.get("x-idempotent-replay") == "true"
        try:
            pred = json.loads(text)
        except ValueError:
            err = "response is not JSON"
        if status != 200 and isinstance(pred, dict) and isinstance(pred.get("error"), dict):
            err = str(pred["error"].get("code"))
    except Exception as e:                                   # noqa: BLE001
        err = f"{type(e).__name__}: {e}"[:300]
    rec["duration_ms"] = round((time.perf_counter() - start) * 1000, 3)
    valid = status == 200 and G.schema_valid(pred, None)
    rec.update(status=status, schema_valid=valid,
               id_echo=valid and pred.get("id") == rec["id"], replay=replay, error=err)
    rec["_pred"] = pred if valid else None
    return rec


async def run(a) -> dict:
    rows = load_inputs(a.inputs)
    if a.limit:
        rows = rows[:a.limit]
    limits = httpx.Limits(max_connections=a.concurrency, max_keepalive_connections=a.concurrency)
    async with httpx.AsyncClient(base_url=a.url, timeout=a.timeout_s, limits=limits) as client:
        health = await wait_ready(client, a.wait_s)
        if health.get("http_status") != 200:
            raise SystemExit(f"service at {a.url} is not ready: {health}")

        async def pump(jobs, t0):
            out, it = [], iter(jobs)

            async def worker():
                for seq, row in it:
                    out.append(await one(client, seq, row, t0, a.timeout_s))

            await asyncio.gather(*(worker() for _ in range(a.concurrency)))
            return out

        warm = [(-1 - i, _warm_row(r, i)) for i, r in enumerate(rows[:a.warmup])]
        if warm:
            await pump(warm, time.perf_counter())
        before = await _get_json(client, "/metrics")
        jobs = [(k * len(rows) + i, r) for k in range(a.repeat) for i, r in enumerate(rows)]
        t0 = time.perf_counter()
        recs = await pump(jobs, t0)
        wall = time.perf_counter() - t0
        after = await _get_json(client, "/metrics")
    recs.sort(key=lambda r: r["seq"])
    return summarise(a, rows, recs, wall, health, before, after)


def _llm(m: dict, key: str) -> float:
    v = (m.get("llm") or {}).get(key)
    return float(v) if isinstance(v, (int, float)) else 0.0


def _llm_delta(before: dict, after: dict, key: str, digits: int = 8):
    """after - before for one GET /metrics llm counter; None when either side lacks it (an older service)."""
    b, c = ((m.get("llm") or {}).get(key) for m in (before, after))
    ok = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (b, c))
    return round(c - b, digits) if ok else None


TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens")


def summarise(a, rows, recs, wall, health, before, after) -> dict:
    done = [r["duration_ms"] for r in recs if r["status"] is not None]
    lat = np.asarray(done) if done else np.zeros(1)
    p50, p95, p99 = (round(float(x), 3) for x in np.percentile(lat, [50, 95, 99]))
    n = len(recs)
    cost = round(_llm(after, "cost_usd") - _llm(before, "cost_usd"), 8)
    uncached = _llm_delta(before, after, "cost_uncached_usd")
    preds = [r.pop("_pred") for r in recs]
    if a.pred_out:
        Path(a.pred_out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.pred_out, "w", encoding="utf-8") as fh:
            for r, p in zip(recs[:len(rows)], preds[:len(rows)]):          # first pass only: one line per id
                if p is not None:
                    fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    return {"label": a.label, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "url": a.url,
            "concurrency": a.concurrency, "sample_count": n, "repeat": a.repeat,
            "errors": sum(r["status"] is None for r in recs),
            "non_200": sum(r["status"] is not None and r["status"] != 200 for r in recs),
            "schema_invalid": sum(r["status"] == 200 and not r["schema_valid"] for r in recs),
            "id_mismatch": sum(r["schema_valid"] and not r["id_echo"] for r in recs),
            "replays": sum(r["replay"] for r in recs),
            "latency_ms": {"p50": p50, "p95": p95, "p99": p99, "max": round(float(lat.max()), 3),
                           "mean": round(float(lat.mean()), 3)},
            "throughput_rps": round(n / wall, 2) if wall > 0 else None, "wall_s": round(wall, 3),
            "client": {"platform": platform.platform(), "python": platform.python_version(),
                       "cpu_count": os.cpu_count()},
            "server_health": health, "server_metrics_before": before, "server_metrics_after": after,
            "llm_calls": int(_llm(after, "calls") - _llm(before, "calls")), "llm_cost_usd": cost,
            "cost_per_message_usd": round(cost / n, 10) if n else None,
            "llm_cost_uncached_usd": uncached,
            "cost_per_message_uncached_usd": round(uncached / n, 10) if n and uncached is not None else None,
            **{f"llm_{k}": _llm_delta(before, after, k, 0) for k in TOKEN_KEYS},
            "requests": recs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--inputs", help="JSONL rows (projected before sending); default dev + dev_noisy + hostile")
    ap.add_argument("--limit", type=int, default=0, help="use the first N rows")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20, help="requests sent first and not recorded")
    ap.add_argument("--out", default=str(ROOT / "results" / "loadtest-latest.json"))
    ap.add_argument("--pred-out", help="write each 200 response body as one JSON line (grader format)")
    ap.add_argument("--label", default="loadtest")
    ap.add_argument("--timeout-s", type=float, default=30.0)
    ap.add_argument("--wait-s", type=float, default=60.0, help="wait this long for GET /healthz to return 200")
    a = ap.parse_args(argv)
    if a.concurrency < 1 or a.repeat < 1 or a.warmup < 0 or a.limit < 0:
        ap.error("--concurrency and --repeat must be at least 1; --warmup and --limit at least 0")
    rep = asyncio.run(run(a))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    keys = ("label", "sample_count", "errors", "non_200", "schema_invalid", "id_mismatch", "replays", "latency_ms",
            "throughput_rps", "wall_s", "llm_calls", "llm_cost_usd", "cost_per_message_usd", "llm_cost_uncached_usd",
            "cost_per_message_uncached_usd", *(f"llm_{k}" for k in TOKEN_KEYS))
    print(json.dumps({k: rep[k] for k in keys}))
    return 1 if rep["errors"] or rep["non_200"] or rep["schema_invalid"] or rep["id_mismatch"] else 0


if __name__ == "__main__":
    sys.exit(main())
