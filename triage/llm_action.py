"""The LLM action step: for rows the CPU model is unsure about, ask an OpenAI-compatible chat endpoint for the next
action with the tested V3 prompt (static system prompt, 8 retrieved train examples, nonce markers, state hint).

One pooled HTTP client, a hard deadline per call, no retries, a circuit breaker, and at most `max_in_flight` calls
waiting at once (more come back "busy" at once, so a slow provider cannot hold every worker thread). Replies over
MAX_REPLY bytes and answers over MAX_CONTENT characters are "invalid", and token counts outside [0, MAX_TOKENS]
give no cost. `decide` never raises: any failure comes back as a status, and the caller keeps the CPU answer. The
API key never appears in a repr, a result or an error text.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from urllib.parse import urlsplit, urlunsplit

import httpx

from . import prompts as P

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-6-luna-20260922"
_LUNA = {"in": 0.10, "out": 0.50, "cache_read": 0.01, "cache_write": 0.125}     # USD per million tokens
PRICES = {"openai/gpt-6-luna-20260922": _LUNA, "openai/gpt-6-luna": _LUNA}       # OpenRouter listing, 13 Oct 2026
STATUSES = ("ok", "timeout", "http_error", "invalid", "abstained", "breaker_open", "busy", "disabled")
FAILURES = frozenset({"timeout", "http_error", "invalid"})                      # these count toward the breaker
MAX_REPLY = 65_536                  # bytes read from the server; a real answer is under 2 KB
MAX_CONTENT = 16_384                # characters of message content parsed
MAX_TOKENS = 10_000_000             # a token count above this, or below 0, is bad data
_TOO_LARGE = object()


def _float(env, name: str, default: float, lo: float, hi: float) -> float:
    raw = env.get(name)
    if raw in (None, ""):
        return default
    try:
        v = float(raw)
    except ValueError:
        v = float("nan")
    if not lo <= v <= hi:
        raise ValueError(f"{name} must be a number from {lo} to {hi}, not {raw!r}")
    return v


@dataclass(frozen=True)
class LLMConfig:
    mode: str = "auto"                       # on | off | auto (on when an API key is set)
    base_url: str = DEFAULT_BASE_URL
    api_key: str = field(default="", repr=False)
    model: str = DEFAULT_MODEL
    provider: str = "openai"                 # OpenRouter host pin; sent only to OpenRouter
    timeout_s: float = 4.0
    gate: float = 0.642841                   # the pipeline calls the step only below this CPU confidence
    breaker_failures: int = 5
    breaker_open_s: float = 30.0
    max_concurrency: int = 32
    max_in_flight: int = 8                   # decide() calls waiting on the server at once; more are "busy"

    @classmethod
    def from_env(cls, env=None) -> "LLMConfig":
        env = os.environ if env is None else env
        mode = (env.get("TRIAGE_LLM") or "auto").strip().lower()
        if mode not in ("on", "off", "auto"):
            raise ValueError(f"TRIAGE_LLM must be on, off or auto, not {mode!r}")
        return cls(mode=mode,
                   base_url=(env.get("TRIAGE_LLM_BASE_URL") or DEFAULT_BASE_URL).strip().rstrip("/"),
                   api_key=(env.get("TRIAGE_LLM_API_KEY") or env.get("OPENROUTER_API_KEY") or "").strip(),
                   model=(env.get("TRIAGE_LLM_MODEL") or DEFAULT_MODEL).strip(),
                   provider=(env.get("TRIAGE_LLM_PROVIDER") or "openai").strip(),
                   timeout_s=_float(env, "TRIAGE_LLM_TIMEOUT_S", 4.0, 0.01, 600.0),
                   gate=_float(env, "TRIAGE_LLM_GATE", 0.642841, 0.0, 1.0))

    @property
    def enabled(self) -> bool:
        return self.mode == "on" or (self.mode == "auto" and bool(self.api_key))

    @property
    def openrouter(self) -> bool:
        return "openrouter.ai" in self.base_url

    def public(self) -> dict:
        """For /healthz and logs: no key, and the base URL without user info or query."""
        try:
            u = urlsplit(self.base_url)
            url = urlunsplit((u.scheme, (u.hostname or "") + (f":{u.port}" if u.port else ""), u.path, "", ""))
        except ValueError:
            url = "(unparsable)"
        return {"enabled": self.enabled, "mode": self.mode, "base_url": url, "key_set": bool(self.api_key),
                "model": self.model, "provider": self.provider if self.openrouter else None,
                "timeout_s": self.timeout_s, "gate": self.gate}


@dataclass(frozen=True)
class LLMResult:
    status: str                              # one of STATUSES
    intent: str | None = None
    action: str | None = None
    confidence: float | None = None          # the model's own figure, as returned
    needs_human: bool | None = None
    latency_s: float = 0.0
    cost_usd: float | None = None            # list price from the token counts
    cost_reported: float | None = None       # usage.cost when the server sends it (OpenRouter)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    model: str | None = None                 # as returned by the server
    provider: str | None = None
    http_status: int | None = None
    error: str | None = None                 # short reason (may be the server's own message); never the key

    @property
    def usable(self) -> bool:
        """True when the action can replace the CPU action."""
        return self.status == "ok"

    def as_dict(self) -> dict:
        return asdict(self)


class CircuitBreaker:
    """Opens after `failures` failures in a row and stays open `open_s` seconds; then one trial call goes
    through: success closes it, failure opens it again. Thread-safe."""

    def __init__(self, failures: int = 5, open_s: float = 30.0, clock=time.monotonic):
        self.failures, self.open_s, self.clock = failures, open_s, clock
        self._lock = threading.Lock()
        self._streak = 0
        self._opened_at: float | None = None
        self._trial = False

    def allow(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return True
            if self._trial or self.clock() - self._opened_at < self.open_s:
                return False
            self._trial = True
            return True

    def record(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self._streak, self._opened_at, self._trial = 0, None, False
                return
            self._streak += 1
            if self._opened_at is None:
                if self._streak >= self.failures:
                    self._opened_at = self.clock()
            elif self._trial:
                self._opened_at, self._trial = self.clock(), False

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "closed"
            return "half_open" if self._trial or self.clock() - self._opened_at >= self.open_s else "open"


def request_body(cfg: LLMConfig, system: str, user: str) -> dict:
    body = {"model": cfg.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "triage", "strict": True, "schema": P.output_schema()}},
            "max_tokens": 2048,
            "reasoning": {"effort": "none"}}
    if cfg.openrouter:
        body["usage"] = {"include": True}
        body["provider"] = {"order": [cfg.provider], "allow_fallbacks": False, "require_parameters": True}
    return body


def list_cost(model: str, prompt: int, completion: int, cached: int, cache_write: int) -> float | None:
    p = PRICES.get(model)
    if p is None:
        return None
    return ((prompt - cached - cache_write) * p["in"] + cached * p["cache_read"] + cache_write * p["cache_write"]
            + completion * p["out"]) / 1e6


def _num(v) -> float | None:
    ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    return float(v) if ok else None


def _count(v) -> int | None:
    """A token count: 0 when missing, None when it is not a number from 0 to MAX_TOKENS."""
    if v is None:
        return 0
    n = _num(v)
    return int(n) if n is not None and 0 <= n <= MAX_TOKENS else None


def _str(v) -> str | None:
    return v if isinstance(v, str) else None


class LLMActionStep:
    """`decide(row)` -> LLMResult for one gate-cleaned row (id, turn_index, context). Safe from many threads.

    Examples come from the retriever's stored payloads, or from `train_by_id` when given."""

    def __init__(self, cfg: LLMConfig, retriever, train_by_id: dict | None = None):
        self.cfg = cfg
        self.retriever = retriever
        self.train_by_id = train_by_id
        self.static = P.static_prompt()
        self.breaker = CircuitBreaker(cfg.breaker_failures, cfg.breaker_open_s)
        self.url = cfg.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
        n = max(1, cfg.max_concurrency)
        self._client = httpx.Client(headers=headers, timeout=httpx.Timeout(cfg.timeout_s),
                                    limits=httpx.Limits(max_connections=n, max_keepalive_connections=n))
        # httpx timeouts are per phase; the pool lets decide() hold a hard deadline over the whole call
        self._pool = cf.ThreadPoolExecutor(max_workers=n, thread_name_prefix="llm")
        self._slots = threading.BoundedSemaphore(max(1, cfg.max_in_flight))

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._client.close()

    def health(self) -> dict:
        return dict(self.cfg.public(), breaker=self.breaker.state)

    def examples(self, row: dict) -> list[dict]:
        if self.train_by_id is not None:
            return [self.train_by_id[i] for i, _ in self.retriever.top_k(row)]
        return self.retriever.examples(row)

    def decide(self, row: dict) -> LLMResult:
        if not self.cfg.enabled:
            return LLMResult("disabled")
        if not self._slots.acquire(blocking=False):                  # taken before the breaker, so a trial call
            return LLMResult("busy")                                   # is never lost to a full house
        try:
            if not self.breaker.allow():
                return LLMResult("breaker_open")
            res = None
            try:
                res = self._decide(row)
            except Exception as e:                                     # a bug here must not cost the CPU answer
                res = LLMResult("invalid", error=f"internal: {type(e).__name__}")
            finally:
                self.breaker.record(res is not None and res.status not in FAILURES)
            return res
        finally:
            self._slots.release()

    def _scrub(self, s) -> str:
        s = str(s)
        return (s.replace(self.cfg.api_key, "[key]") if self.cfg.api_key else s)[:200]

    def _post(self, body: dict) -> tuple[int, object]:
        raw = bytearray()
        with self._client.stream("POST", self.url, json=body) as r:
            for chunk in r.iter_bytes():
                raw += chunk
                if len(raw) > MAX_REPLY:
                    return r.status_code, _TOO_LARGE
        try:
            return r.status_code, json.loads(raw)
        except ValueError:
            return r.status_code, raw[:2000].decode("utf-8", "replace")

    def _decide(self, row: dict) -> LLMResult:
        user, _ = P.dynamic_prompt(row, self.examples(row))
        body = request_body(self.cfg, self.static, user)
        t0 = time.perf_counter()
        fut = self._pool.submit(self._post, body)
        try:
            code, data = fut.result(timeout=self.cfg.timeout_s)
        except (cf.TimeoutError, httpx.TimeoutException):
            fut.cancel()
            return LLMResult("timeout", latency_s=time.perf_counter() - t0, error="timeout")
        except httpx.HTTPError as e:
            return LLMResult("http_error", latency_s=time.perf_counter() - t0, error=self._scrub(type(e).__name__))
        lat = time.perf_counter() - t0
        if data is _TOO_LARGE:
            return LLMResult("invalid", latency_s=lat, http_status=code, error=f"reply over {MAX_REPLY} bytes")

        if code >= 400 or not isinstance(data, dict) or data.get("error"):
            err = data.get("error") if isinstance(data, dict) else data
            if isinstance(err, dict):
                err = err.get("message") or err
            return LLMResult("http_error", latency_s=lat, http_status=code, error=self._scrub(err or "bad body"))

        u = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        det = u.get("prompt_tokens_details") if isinstance(u.get("prompt_tokens_details"), dict) else {}
        counts = [_count(u.get("prompt_tokens")), _count(u.get("completion_tokens")),
                  _count(det.get("cached_tokens")), _count(det.get("cache_write_tokens"))]
        sane = None not in counts and counts[2] + counts[3] <= counts[0]
        pt, ct, cr, cw = (c or 0 for c in counts)
        model = _str(data.get("model"))
        reported = _num(u.get("cost"))
        meta = dict(latency_s=lat, http_status=code, model=model, provider=_str(data.get("provider")),
                    prompt_tokens=pt, completion_tokens=ct, cached_tokens=cr, cache_write_tokens=cw,
                    cost_usd=list_cost(self.cfg.model if self.cfg.model in PRICES else model, pt, ct, cr, cw)
                    if sane else None,
                    cost_reported=reported if reported is not None and reported >= 0 else None)

        choices = data.get("choices")
        first = choices[0] if isinstance(choices, list) and choices else None
        msg = first.get("message") if isinstance(first, dict) else None
        text = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(text, str) and len(text) > MAX_CONTENT:
            return LLMResult("invalid", error=f"answer over {MAX_CONTENT} characters", **meta)
        pred, pstatus = P.parse_output(text) if text is None or isinstance(text, str) else (None, "invalid_json")
        if pstatus not in ("ok", "lenient_ok"):
            return LLMResult("invalid", error=pstatus, **meta)
        status = "abstained" if pred["intent"] == "unknown" or pred["action"] == "none" else "ok"
        return LLMResult(status, intent=pred["intent"], action=pred["action"], confidence=pred["confidence"],
                         needs_human=pred["needs_human"], **meta)
