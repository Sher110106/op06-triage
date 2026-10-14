"""HTTP service: POST /triage, GET /healthz, GET /metrics.

Transport only: body limit (413), body-read deadline (408), load shedding (503), strict JSON and the envelope
check (400), request ids, an idempotency cache, metrics and one JSON log line per request. The work happens in
`Triage.handle` on a worker thread; large bodies are parsed on a worker thread too.

  uvicorn triage.service:app --host 0.0.0.0 --port $PORT --workers 1 --limit-concurrency 256 --timeout-keep-alive 5 \
      --no-access-log
"""
from __future__ import annotations

import asyncio
import concurrent.futures as cf
import hashlib
import json
import logging
import math
import os
import re
import sys
import threading
import time
import uuid
from collections import Counter, OrderedDict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import Response
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect

from .gate import envelope_error

MAX_BODY = 1_048_576
BODY_TIMEOUT_S = 10.0               # the whole body must arrive within this (408 after)
MAX_PENDING = 128                   # POST /triage requests in the app at once; more get a JSON 503 at once
IDEM_SIZE = 10_000
DEGRADED_TTL_S = 30.0               # answers made while the LLM step failed stay cached this long (the breaker's window)
MAX_DEPTH = 64                      # nesting levels; a valid request needs 3
WORKERS = 16                        # handle() calls running at once
PARSE_INLINE = 16_384               # bodies up to this size are parsed on the event loop; larger ones on a thread
PARSE_WORKERS = 2                   # large bodies parsed at once, so they cannot fill the event loop's thread pool
LATENCY_WINDOW = 10_000
LLM_CALLED = frozenset({"ok", "timeout", "http_error", "invalid", "abstained"})    # statuses where a call went out
DEGRADED = frozenset({"timeout", "http_error", "invalid", "breaker_open", "busy"})   # CPU answer kept for these
LOG_KEYS = ("flags", "route_reasons", "llm_status", "llm_latency_ms", "llm_cost_usd", "policy_version",
            "needs_human", "abstained", "replay", "error", "error_type")
_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")

log = logging.getLogger("triage.service")


class BadJSON(ValueError):
    pass


def _json(obj) -> bytes:
    """ASCII JSON: any id echoes back, a lone surrogate included (as \\ud800)."""
    return json.dumps(obj, allow_nan=False, separators=(",", ":"), default=str).encode("ascii")


def _error(code: str, message: str) -> bytes:
    return _json({"error": {"code": code, "message": message}})


def abstention(rid: str, code: str, message: str) -> dict:
    return {"id": rid, "intent": "unknown", "action": "none", "confidence": 0.0, "needs_human": True,
            "error": {"code": code, "message": message}}


def request_id(header: str | None) -> str:
    """The caller's X-Request-ID when it is 1-128 characters of [A-Za-z0-9._:-], else a new uuid4 hex."""
    return header if header is not None and _REQUEST_ID.fullmatch(header) else uuid.uuid4().hex


def _no_constant(name: str):
    raise BadJSON(f"{name} is not allowed in JSON")


def _depth_ok(obj, limit: int) -> bool:
    level = [obj]
    for _ in range(limit):
        nxt = []
        for v in level:
            if isinstance(v, dict):
                nxt.extend(c for c in v.values() if isinstance(c, (dict, list)))
            elif isinstance(v, list):
                nxt.extend(c for c in v if isinstance(c, (dict, list)))
        if not nxt:
            return True
        level = nxt
    return False


def parse_body(raw: bytes) -> tuple[object, str]:
    """(parsed body, SHA-256 of its canonical form). Raises BadJSON with a short message."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise BadJSON("body is not valid UTF-8") from None
    try:
        obj = json.loads(text, parse_constant=_no_constant)
    except BadJSON:
        raise
    except json.JSONDecodeError as e:
        raise BadJSON(f"body is not valid JSON: {e.msg} at character {e.pos}") from None
    except RecursionError:
        raise BadJSON(f"JSON is nested deeper than {MAX_DEPTH} levels") from None
    except ValueError:                                  # an integer over Python's 4,300-digit limit
        raise BadJSON("a number in the body is too long") from None
    if not _depth_ok(obj, MAX_DEPTH):
        raise BadJSON(f"JSON is nested deeper than {MAX_DEPTH} levels")
    canon = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return obj, hashlib.sha256(canon.encode("utf-8", "surrogatepass")).hexdigest()


def _parse(raw: bytes) -> tuple[object, str, str | None]:
    obj, key = parse_body(raw)
    return obj, key, envelope_error(obj)


def _num(v) -> float | None:
    ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    return float(v) if ok else None


def valid_response(resp, rid: str) -> bool:
    """The checks of the grader's schema_valid, plus the id echo."""
    if not isinstance(resp, dict) or resp.get("id") != rid:
        return False
    if not (isinstance(resp.get("intent"), str) and isinstance(resp.get("action"), str)
            and isinstance(resp.get("needs_human"), bool)):
        return False
    c = _num(resp.get("confidence"))
    if c is None or not 0.0 <= c <= 1.0:
        return False
    return all(_num(resp[k]) is not None and resp[k] >= 0 for k in ("cost_usd", "latency_ms") if k in resp)


def _strs(v) -> list[str]:
    return [str(x) for x in v] if isinstance(v, (list, tuple, set, frozenset)) else []


def trace_fields(trace) -> dict:
    """The parts of a pipeline trace that go to logs and metrics. Tolerant of missing or odd values."""
    t = trace if isinstance(trace, dict) else {}
    llm = t.get("llm") if isinstance(t.get("llm"), dict) else {}
    status = llm.get("status", t.get("llm_status"))
    lat = _num(llm.get("latency_s"))
    lat_ms = round(lat * 1000, 1) if lat is not None else _num(t.get("llm_latency_ms"))
    pv = t.get("policy_version")
    return {"flags": sorted(_strs(t.get("flags"))), "route_reasons": _strs(t.get("route_reasons")),
            "llm_status": status if isinstance(status, str) else None, "llm_latency_ms": lat_ms,
            "llm_cost_usd": _num(llm.get("cost_usd", t.get("llm_cost_usd"))),
            "policy_version": pv if isinstance(pv, str) else None,
            "error_type": t["error"] if isinstance(t.get("error"), str) else None}


def percentiles(values, qs=(50, 95, 99)) -> dict:
    v = sorted(values)
    out = {}
    for q in qs:
        if not v:
            out[f"p{q}"] = None
            continue
        k = (len(v) - 1) * q / 100
        lo = math.floor(k)
        hi = min(lo + 1, len(v) - 1)
        out[f"p{q}"] = round(v[lo] + (v[hi] - v[lo]) * (k - lo), 3)
    return out


@dataclass(frozen=True)
class Answer:
    body: bytes
    info: dict = field(default_factory=dict)        # trace fields, needs_human, abstained, error
    cacheable: bool = True
    ttl_s: float | None = None                      # cache lifetime; None: until evicted or the policy changes


def _load_from_env():
    from .pipeline import Triage
    return Triage.from_env()


def _setup_logging() -> None:
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(h)
    log.setLevel(logging.INFO)
    log.propagate = False


def _event(**kv) -> None:
    log.info(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **kv},
                        separators=(",", ":"), default=str))


class Service:
    """State behind one app: the Triage object, the idempotency cache and the metrics."""

    def __init__(self, triage, loader, max_body: int, idem_size: int, body_timeout_s: float = BODY_TIMEOUT_S,
                 max_pending: int = MAX_PENDING, degraded_ttl_s: float = DEGRADED_TTL_S,
                 exit_on_load_error: bool = False):
        self.triage = triage
        self.loader = loader or _load_from_env
        self.load_error: str | None = None
        self.load_message: str | None = None
        self.exit_on_load_error = exit_on_load_error
        self.owned = False
        self.max_body, self.idem_size = max_body, idem_size
        self.body_timeout_s, self.max_pending, self.degraded_ttl_s = body_timeout_s, max_pending, degraded_ttl_s
        self.pending = 0                                 # POST /triage requests in the app; event loop only
        self.started = time.monotonic()
        self.limiter: anyio.CapacityLimiter | None = None
        self.parse_limiter: anyio.CapacityLimiter | None = None
        self.lock = threading.Lock()                     # cache, in-flight table, policy
        # (generation, body sha) -> (answer, expiry or None); a new policy starts a new generation
        self.cache: OrderedDict[tuple[int, str], tuple[Answer, float | None]] = OrderedDict()
        self.inflight: dict[tuple[int, str], cf.Future] = {}
        self.policy: object | None = None                # the Policy object the cached answers were made under
        self.policy_version: str | None = None
        self.gen = 0
        self.mlock = threading.Lock()                    # metrics
        self.latency: deque[float] = deque(maxlen=LATENCY_WINDOW)
        self.count: Counter = Counter()
        self.by_status: Counter = Counter()
        self.errors: Counter = Counter()
        self.flags: Counter = Counter()
        self.reasons: Counter = Counter()
        self.llm_status: Counter = Counter()
        self.llm_cost = 0.0

    # ------------------------------------------------------------------ life cycle

    def start(self) -> None:
        if self.limiter is None:
            self.limiter = anyio.CapacityLimiter(WORKERS)
        if self.parse_limiter is None:
            self.parse_limiter = anyio.CapacityLimiter(PARSE_WORKERS)
        if self.triage is None and self.load_error is None:
            threading.Thread(target=self._load, name="triage-load", daemon=True).start()

    def _load(self) -> None:
        t0 = time.perf_counter()
        try:
            tr = self.loader()
        except BaseException as e:                       # noqa: BLE001 - any failure is reported, not raised
            self.load_error, self.load_message = type(e).__name__, str(e)[:300]
            _event(event="load_failed", error_type=self.load_error, message=self.load_message)
            if self.exit_on_load_error:                  # a bad setting or file: stop, so a supervisor sees it
                sys.stdout.flush()
                os._exit(3)
            return
        self.owned = True
        self.triage = tr
        _event(event="loaded", load_s=round(time.perf_counter() - t0, 3))

    def close(self) -> None:
        if self.owned and hasattr(self.triage, "close"):
            try:
                self.triage.close()
            except Exception:                            # noqa: BLE001
                pass

    def uptime(self) -> float:
        return round(time.monotonic() - self.started, 3)

    # ------------------------------------------------------------------ POST /triage

    async def triage_request(self, request: Request, rid: str) -> tuple[int, bytes, dict]:
        if self.triage is None:
            code = "unavailable" if self.load_error else "loading"
            msg = "the model failed to load" if self.load_error else "the model is still loading"
            return 503, _error(code, msg), {"error": code}
        cl = request.headers.get("content-length")
        if cl is not None:
            if not (cl.isascii() and cl.isdigit()):
                return 400, _error("bad_request", "Content-Length is not a number"), {"error": "bad_request"}
            if int(cl) > self.max_body:
                return 413, self._too_large(), {"error": "body_too_large"}
        raw = bytearray()
        try:
            with anyio.fail_after(self.body_timeout_s):
                async for chunk in request.stream():
                    raw += chunk
                    if len(raw) > self.max_body:
                        return 413, self._too_large(), {"error": "body_too_large"}
        except ClientDisconnect:
            return 499, b"", {"error": "client_disconnect"}
        except TimeoutError:
            msg = f"the body did not arrive within {self.body_timeout_s:g} s"
            return 408, _error("body_timeout", msg), {"error": "body_timeout"}
        try:
            if len(raw) <= PARSE_INLINE:
                obj, key, msg = _parse(bytes(raw))
            else:                                        # up to 1 MiB of JSON: keep the event loop free
                if self.parse_limiter is None:
                    self.parse_limiter = anyio.CapacityLimiter(PARSE_WORKERS)
                obj, key, msg = await anyio.to_thread.run_sync(_parse, bytes(raw), limiter=self.parse_limiter)
        except BadJSON as e:
            return 400, _error("bad_json", str(e)), {"error": "bad_json"}
        if msg is not None:
            return 400, _error("invalid_envelope", msg), {"error": "invalid_envelope"}
        ans, replay = await self._answer(obj, key, rid)
        return 200, ans.body, {**ans.info, "replay": replay}

    def _too_large(self) -> bytes:
        return _error("body_too_large", f"body is over {self.max_body} bytes")

    def _generation(self) -> int:
        """The cache generation for the policy in force. A reload of the policy file (with or without a new version)
        gives the store a new Policy object; that starts a new generation and drops every older answer."""
        store = getattr(self.triage, "store", None)
        try:
            policy = store.current() if store is not None else None
        except Exception:                                # noqa: BLE001 - keep serving with the last known policy
            policy = None
        with self.lock:
            if policy is not None and policy is not self.policy:
                if self.policy is not None:
                    self.cache.clear()
                    self.gen += 1
                self.policy = policy
                self.policy_version = getattr(policy, "version", None)
            return self.gen

    def _cached(self, key: tuple[int, str]) -> Answer | None:
        """Call with self.lock held."""
        item = self.cache.get(key)
        if item is None:
            return None
        ans, expires = item
        if expires is not None and time.monotonic() >= expires:
            del self.cache[key]
            return None
        self.cache.move_to_end(key)
        return ans

    async def _answer(self, obj: dict, body_key: str, rid: str) -> tuple[Answer, bool]:
        """A cached answer, the answer of an identical request in flight, or a new one. handle() runs at most
        once per body and policy while its answer stays cached."""
        key = (self._generation(), body_key)
        with self.lock:
            ans = self._cached(key)
            if ans is not None:
                return ans, True
            fut = self.inflight.get(key)
            mine = fut is None
            if mine:
                fut = self.inflight[key] = cf.Future()
        if not mine:
            ans = await asyncio.shield(asyncio.wrap_future(fut))
            if ans is not None:
                return ans, True
            ans = await self._compute(obj, rid)          # the first request failed: try again, no waiting
            self._put(key, ans)
            return ans, False
        ans = None
        try:
            ans = await self._compute(obj, rid)
            self._put(key, ans)
        finally:
            with self.lock:
                self.inflight.pop(key, None)
            if not fut.done():
                fut.set_result(ans if ans is not None and ans.cacheable else None)
        return ans, False

    def _put(self, key: tuple[int, str], ans: Answer) -> None:
        if not ans.cacheable or self.idem_size <= 0 or (ans.ttl_s is not None and ans.ttl_s <= 0):
            return
        expires = None if ans.ttl_s is None else time.monotonic() + ans.ttl_s
        with self.lock:
            if key[0] != self.gen:                       # made under a policy that has since been replaced
                return
            self.cache[key] = (ans, expires)
            self.cache.move_to_end(key)
            while len(self.cache) > self.idem_size:
                self.cache.popitem(last=False)

    async def _compute(self, obj: dict, rid: str) -> Answer:
        if self.limiter is None:
            self.limiter = anyio.CapacityLimiter(WORKERS)
        return await anyio.to_thread.run_sync(self._run, obj, rid, limiter=self.limiter)

    def _run(self, obj: dict, rid: str) -> Answer:
        row_id = obj["id"]
        try:
            resp, trace = self.triage.handle(obj, rid)
            if not valid_response(resp, row_id):
                raise ValueError("handle returned a response that does not match the schema")
            body = _json(resp)
        except Exception as e:                           # noqa: BLE001 - one bad request never takes the service down
            resp = abstention(row_id, "internal", "internal error")
            return Answer(_json(resp), {"needs_human": True, "abstained": True, "error": "internal",
                                        "error_type": type(e).__name__}, cacheable=False)
        info = {**trace_fields(trace), "needs_human": resp["needs_human"], "abstained": "error" in resp}
        if info["error_type"]:                           # handle() caught its own failure: answer, but do not cache
            return Answer(body, {**info, "error": "internal"}, cacheable=False)
        if info["llm_status"] in DEGRADED:               # the LLM step failed: a retry soon may get a better answer
            return Answer(body, info, ttl_s=self.degraded_ttl_s)
        return Answer(body, info)

    # ------------------------------------------------------------------ metrics and logs

    def record(self, status: int, ms: float, info: dict) -> None:
        with self.mlock:
            self.count["requests"] += 1
            self.by_status[str(status)] += 1
            self.latency.append(ms)
            if info.get("error"):
                self.errors[info["error"]] += 1
            if status != 200:
                return
            self.count["routed"] += bool(info.get("needs_human"))
            self.count["abstentions"] += bool(info.get("abstained"))
            if info.get("replay"):
                self.count["replays"] += 1
                return
            self.flags.update(info.get("flags") or ())
            self.reasons.update(info.get("route_reasons") or ())
            st = info.get("llm_status")
            if st:
                self.llm_status[st] += 1
            cost = info.get("llm_cost_usd")
            if isinstance(cost, float) and 0 < cost < 1.0:   # one call costs well under a cent; anything else is bad data
                self.llm_cost += cost

    def metrics(self) -> dict:
        with self.mlock:
            lat = list(self.latency)
            out = {"uptime_s": self.uptime(), "ready": self.triage is not None,
                   "requests": self.count["requests"], "by_status": dict(self.by_status),
                   "errors": dict(self.errors), "routed": self.count["routed"],
                   "abstentions": self.count["abstentions"], "replays": self.count["replays"],
                   "flags": dict(self.flags), "route_reasons": dict(self.reasons),
                   "llm": {"calls": sum(n for s, n in self.llm_status.items() if s in LLM_CALLED),
                           "by_status": dict(self.llm_status), "cost_usd": round(self.llm_cost, 8)},
                   "latency_ms": {"window": len(lat), **percentiles(lat)}}
        with self.lock:
            out["idempotency"] = {"entries": len(self.cache), "capacity": self.idem_size,
                                  "in_flight": len(self.inflight)}
            out["policy_version"] = self.policy_version
        return out


def _log_request(rid: str, status: int, ms: float, info: dict) -> None:
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "request_id": rid,
           "status": status, "latency_ms": round(ms, 2)}
    rec.update({k: info.get(k) for k in LOG_KEYS})
    if info.get("replay"):                             # no LLM call on a replay
        rec.update(llm_status=None, llm_latency_ms=None, llm_cost_usd=None)
    log.info(json.dumps(rec, separators=(",", ":"), default=str))


def _respond(status: int, body: bytes, headers: dict | None = None) -> Response:
    return Response(body, status_code=status, media_type="application/json", headers=headers)


def create_app(triage=None, *, max_body: int = MAX_BODY, idem_size: int = IDEM_SIZE, loader=None,
               body_timeout_s: float = BODY_TIMEOUT_S, max_pending: int = MAX_PENDING,
               degraded_ttl_s: float = DEGRADED_TTL_S, exit_on_load_error: bool = False) -> FastAPI:
    """The app. With triage=None the model loads on a background thread at start-up (`loader`, by default
    Triage.from_env()); /healthz answers 503 until it is ready. If the load fails, /healthz answers 503 with the
    reason, or with exit_on_load_error the process exits with status 3."""
    _setup_logging()
    svc = Service(triage, loader, max_body, idem_size, body_timeout_s, max_pending, degraded_ttl_s,
                  exit_on_load_error)

    @asynccontextmanager
    async def lifespan(app):
        svc.start()
        yield
        svc.close()

    app = FastAPI(title="OP-06 triage", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.service = svc

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _respond(exc.status_code, _error(code, str(exc.detail)), exc.headers)

    @app.exception_handler(Exception)
    async def server_error(request: Request, exc: Exception):
        return _respond(500, _error("server_error", "unexpected server error"))

    @app.get("/healthz")
    async def healthz():
        if svc.triage is None:
            if svc.load_error:
                return _respond(503, _json({"status": "error", "error": svc.load_error, "message": svc.load_message,
                                            "uptime_s": svc.uptime()}))
            return _respond(503, _json({"status": "loading", "uptime_s": svc.uptime()}), {"Retry-After": "1"})
        try:
            h = await anyio.to_thread.run_sync(svc.triage.health)
            body = _json({"status": "ok", **(h if isinstance(h, dict) else {}), "uptime_s": svc.uptime()})
        except Exception as e:                           # noqa: BLE001 - the model is loaded; say so
            body = _json({"status": "ok", "health_error": type(e).__name__, "uptime_s": svc.uptime()})
        return _respond(200, body)

    @app.get("/metrics")
    async def metrics():
        return _respond(200, _json(svc.metrics()))

    @app.post("/triage")
    async def post_triage(request: Request):
        t0 = time.perf_counter()
        rid = request_id(request.headers.get("x-request-id"))
        if svc.pending >= svc.max_pending:               # shed here, as JSON that /metrics and the log both see
            status, body, info = 503, _error("overloaded", "too many requests at once; retry shortly"), \
                {"error": "overloaded"}
        else:
            svc.pending += 1
            try:
                status, body, info = await svc.triage_request(request, rid)
            except Exception as e:                       # noqa: BLE001
                status, body = 500, _error("server_error", "unexpected server error")
                info = {"error": "server_error", "error_type": type(e).__name__}
            finally:
                svc.pending -= 1
        ms = (time.perf_counter() - t0) * 1000
        svc.record(status, ms, info)
        _log_request(rid, status, ms, info)
        headers = {"X-Request-ID": rid}
        if info.get("replay"):
            headers["X-Idempotent-Replay"] = "true"
        if info.get("error") in ("overloaded", "loading"):
            headers["Retry-After"] = "1"
        if status == 408:                                # the rest of the body may never come
            headers["Connection"] = "close"
        return _respond(status, body, headers)

    return app


app = create_app(exit_on_load_error=True)
