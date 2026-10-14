"""Tests for triage/service.py with a fake Triage (fast), plus one real uvicorn run when the model is present."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
import shutil
from functools import cache
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from triage.data import DATA, ROOT, project
from triage import service as S
from triage.service import MAX_BODY, MAX_DEPTH, PARSE_INLINE, create_app, parse_body, percentiles, request_id

BUNDLE = ROOT / "models" / "classical_v0.joblib"
MARKER = "zq7xmarker-do-not-log"
ROW = {"id": "row-1", "turn_index": 3, "context": [
    {"speaker": "customer", "text": "hi, my order never arrived"},
    {"speaker": "agent", "text": "sorry to hear that, let me look"},
    {"speaker": "action", "text": "pull-up-account"}]}


@cache
def grader():
    spec = importlib.util.spec_from_file_location("op06_grader", DATA / "grader.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def schema_valid(pred) -> bool:
    return grader().schema_valid(pred, None)


class FakeStore:
    """Like policy.PolicyStore: current() returns the Policy object in force; a reload swaps in a new object."""

    def __init__(self, version: str):
        self.policy = SimpleNamespace(version=version)

    def current(self):
        return self.policy

    def reload(self, version: str | None = None):
        self.policy = SimpleNamespace(version=version or self.policy.version)


class FakeTriage:
    """handle() returns a fixed valid answer and counts calls; `delay` slows it, `fail` makes it raise.
    The answer depends on the policy in force: `conf` maps a policy version to a confidence."""

    def __init__(self, delay: float = 0.0, fail: bool = False, response: dict | None = None,
                 llm_status: str = "ok"):
        self.delay, self.fail, self.response, self.llm_status = delay, fail, response, llm_status
        self.calls, self.request_ids, self.keys = 0, [], set()
        self.store = FakeStore("test.1")
        self.conf = {}
        self.lock = threading.Lock()

    @property
    def policy_version(self):
        return self.store.policy.version

    def handle(self, obj, request_id):
        with self.lock:
            self.calls += 1
            self.request_ids.append(request_id)
            self.keys.update(obj)
        policy = self.store.current()
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("boom " + MARKER)
        if self.response is not None:
            return {**self.response, "id": obj["id"]}, {}
        routed = "route me" in json.dumps(obj)
        resp = {"id": obj["id"], "intent": "status_delivery_time", "action": "shipping-status",
                "confidence": 0.31 if routed else self.conf.get(id(policy), 0.82), "needs_human": routed}
        trace = {"flags": ["empty_turns"] + (["second_intent"] if routed else []),
                 "route_reasons": ["low_confidence"] if routed else [],
                 "llm": {"status": self.llm_status, "latency_s": 0.25, "cost_usd": 0.0001} if routed else None,
                 "policy_version": policy.version}
        return resp, trace

    def health(self):
        return {"bundle_sha256": "f" * 64, "policy": {"version": self.policy_version}}


@pytest.fixture
def make_client():
    opened = []

    def make(triage=None, **kw):
        tr = triage if triage is not None else FakeTriage()
        c = TestClient(create_app(tr, **kw))
        c.__enter__()
        c.triage = tr
        opened.append(c)
        return c

    yield make
    for c in opened:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    return make_client()


def post(c, body, **kw):
    if not isinstance(body, (bytes, str)):
        body = json.dumps(body)
    return c.post("/triage", content=body, headers={"content-type": "application/json", **kw.pop("headers", {})}, **kw)


# ------------------------------------------------------------------ /healthz and loading

def test_healthz_loading_then_ready():
    release = threading.Event()
    fake = FakeTriage()

    def loader():
        release.wait(10)
        return fake

    with TestClient(create_app(loader=loader)) as c:
        r = c.get("/healthz")
        assert r.status_code == 503 and r.json()["status"] == "loading"
        assert r.headers["content-type"] == "application/json"
        r = post(c, ROW)
        assert r.status_code == 503 and r.json()["error"]["code"] == "loading"
        release.set()
        deadline = time.monotonic() + 5
        while (r := c.get("/healthz")).status_code != 200 and time.monotonic() < deadline:
            time.sleep(0.01)
        body = r.json()
        assert r.status_code == 200 and body["status"] == "ok"
        assert body["bundle_sha256"] == "f" * 64 and body["uptime_s"] >= 0
        assert post(c, ROW).status_code == 200


def test_load_failure_is_503():
    def loader():
        raise FileNotFoundError("models/missing.joblib")

    with TestClient(create_app(loader=loader)) as c:
        deadline = time.monotonic() + 5
        while (r := c.get("/healthz")).json()["status"] == "loading" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert r.status_code == 503 and r.json()["status"] == "error" and r.json()["error"] == "FileNotFoundError"
        assert r.json()["message"] == "models/missing.joblib"           # the reason, not only the type
        r = post(c, ROW)
        assert r.status_code == 503 and r.json()["error"]["code"] == "unavailable"


def test_load_failure_exits_when_asked(monkeypatch):
    """The uvicorn app exits with status 3 on a bad setting, so a supervisor or deploy sees the failure."""
    codes, done = [], threading.Event()

    def fake_exit(code):
        codes.append(code)
        done.set()

    def loader():
        raise ValueError("TRIAGE_LLM must be on, off or auto, not 'maybe'")

    monkeypatch.setattr(S.os, "_exit", fake_exit)
    with TestClient(create_app(loader=loader, exit_on_load_error=True)):
        assert done.wait(5)
    assert codes == [3]
    assert S.app.state.service.exit_on_load_error is True                 # the app uvicorn serves


def test_healthz_survives_health_error(make_client):
    fake = FakeTriage()
    fake.health = lambda: 1 / 0
    r = make_client(fake).get("/healthz")
    assert r.status_code == 200 and r.json()["health_error"] == "ZeroDivisionError"


# ------------------------------------------------------------------ POST /triage: good requests

def test_valid_post(client):
    r = post(client, ROW)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/json"
    body = r.json()
    assert schema_valid(body) and body["id"] == "row-1"
    assert client.triage.calls == 1


def test_hostile_rows_pass_through(client, split):
    for row in split("hostile"):
        r = post(client, project(row))
        assert r.status_code == 200 and schema_valid(r.json()) and r.json()["id"] == row["id"]


def test_extra_fields_and_unicode_are_fine(client):
    row = {**ROW, "id": "café-\U0001f600", "extra": {"nested": [1, 2.5, None, True]},
           "context": [{"speaker": "robot", "text": "\ud800 lone surrogate"}]}
    r = post(client, json.dumps(row))                # ensure_ascii escapes the surrogate, as a client would
    assert r.status_code == 200 and r.json()["id"] == row["id"]


@pytest.mark.parametrize("rid", ["x" * 300, "x" * 100_000, "\ud800", "a\udfffb", "caf\u00e9-\U0001f600"])
def test_any_non_empty_id_is_answered(client, rid):
    """input.schema.json asks only for a non-empty string id; the response echoes it, lone surrogates included."""
    r = post(client, json.dumps({**ROW, "id": rid}))
    assert r.status_code == 200 and r.content.isascii()
    assert r.json()["id"] == rid and schema_valid(r.json())


# ------------------------------------------------------------------ POST /triage: bad requests

BAD_JSON = {
    "malformed": b'{"id": "x", "context": [',
    "empty": b"",
    "nan": b'{"id": "x", "context": [], "turn_index": NaN}',
    "infinity": b'{"id": "x", "context": [], "turn_index": Infinity}',
    "minus_infinity": b'{"id": "x", "context": [], "turn_index": -Infinity}',
    "nested_2000": b'{"id": "x", "context": [], "deep": ' + b"[" * 2000 + b"]" * 2000 + b"}",
    "nested_200k": b"[" * 200_000 + b"]" * 200_000,
    "invalid_utf8": b'{"id": "x\xff\xfe", "context": []}',
    "utf8_bom": b'\xef\xbb\xbf{"id": "x", "context": []}',
    "trailing_garbage": b'{"id": "x", "context": []} extra',
    "huge_int": b'{"id": "x", "context": [], "turn_index": 1' + b"0" * 5000 + b"}",
}
BAD_ENVELOPE = {
    "top_level_list": b'[{"id": "x", "context": []}]',
    "top_level_string": b'"hello"',
    "missing_id": b'{"context": []}',
    "empty_id": b'{"id": "", "context": []}',
    "id_int": b'{"id": 7, "context": []}',
    "id_null": b'{"id": null, "context": []}',
}


@pytest.mark.parametrize("name", sorted(BAD_JSON))
def test_bad_json_is_400(client, name):
    r = post(client, BAD_JSON[name])
    assert r.status_code == 400 and r.headers["content-type"] == "application/json"
    assert r.json()["error"]["code"] == "bad_json" and r.json()["error"]["message"]
    assert client.triage.calls == 0


@pytest.mark.parametrize("name", sorted(BAD_ENVELOPE))
def test_bad_envelope_is_400(client, name):
    r = post(client, BAD_ENVELOPE[name])
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_envelope"
    assert client.triage.calls == 0


def test_nesting_limit_edge():
    ok = {"id": "x", "context": []}                  # two levels
    for _ in range(MAX_DEPTH - 2):
        ok = {"a": ok}
    parse_body(json.dumps(ok).encode())
    with pytest.raises(ValueError):
        parse_body(json.dumps({"a": ok}).encode())


def _sized_body(n: int) -> bytes:
    head = json.dumps({**ROW, "context": [{"speaker": "customer", "text": ""}]}).encode()
    body = head.replace(b'"text": ""', b'"text": "' + b"a" * (n - len(head)) + b'"')
    assert len(body) == n
    return body


def test_body_at_limit_is_accepted(client):
    assert post(client, _sized_body(MAX_BODY)).status_code == 200


def test_body_over_limit_with_content_length(client):
    r = post(client, _sized_body(MAX_BODY + 1))
    assert r.status_code == 413 and r.json()["error"]["code"] == "body_too_large"
    assert client.triage.calls == 0


def test_body_over_limit_chunked(client):
    body = _sized_body(MAX_BODY + 10)

    def chunks():
        for i in range(0, len(body), 65536):
            yield body[i:i + 65536]

    r = client.post("/triage", content=chunks(), headers={"content-type": "application/json"})
    assert r.status_code == 413 and r.json()["error"]["code"] == "body_too_large"
    assert client.triage.calls == 0


def test_chunked_body_under_limit(client):
    r = client.post("/triage", content=iter([json.dumps(ROW).encode()]))
    assert r.status_code == 200 and r.json()["id"] == "row-1"


def test_bad_content_length(client):
    r = client.post("/triage", content=b"{}", headers={"content-length": "abc"})
    assert r.status_code == 400


async def _asgi_post(app, receive, path="/triage"):
    """Call the ASGI app directly with a custom receive(); returns (status, headers, body)."""
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "",
             "headers": [(b"content-type", b"application/json"), (b"content-length", b"100")],
             "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 80)}
    sent = []

    async def send(msg):
        sent.append(msg)

    await app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    return start["status"], dict(start["headers"]), b"".join(m.get("body", b"") for m in sent
                                                             if m["type"] == "http.response.body")


def test_stalled_body_gets_408():
    """A client that sends 6 of 100 promised body bytes and then stalls gets a 408, not a slot held forever."""
    app = create_app(FakeTriage(), body_timeout_s=0.3)
    parts = [{"type": "http.request", "body": b'{"id":', "more_body": True}]

    async def receive():
        if parts:
            return parts.pop(0)
        await asyncio.sleep(3600)

    async def run():
        async with app.router.lifespan_context(app):
            t0 = time.perf_counter()
            out = await _asgi_post(app, receive)
            return out, time.perf_counter() - t0

    (status, headers, body), wall = asyncio.run(run())
    assert status == 408 and json.loads(body)["error"]["code"] == "body_timeout"
    assert headers[b"connection"] == b"close" and 0.25 < wall < 2
    assert app.state.service.metrics()["errors"] == {"body_timeout": 1}


def test_overload_sheds_with_json_503(make_client):
    """Past max_pending requests in the app, POST /triage answers a JSON 503 at once; /metrics counts it and
    /healthz still answers."""
    release = threading.Event()

    class Slow(FakeTriage):
        def handle(self, obj, request_id):
            release.wait(10)
            return super().handle(obj, request_id)

    c = make_client(Slow(), max_pending=2)
    out = []
    threads = [threading.Thread(target=lambda i=i: out.append(post(c, {**ROW, "id": f"s{i}"}))) for i in range(2)]
    for t in threads:
        t.start()
    deadline = time.monotonic() + 5
    while c.app.state.service.pending < 2:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    r = post(c, {**ROW, "id": "shed"}, headers={"X-Request-ID": "shed-1"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "overloaded"
    assert r.headers["retry-after"] == "1" and r.headers["x-request-id"] == "shed-1"
    assert c.get("/healthz").status_code == 200
    release.set()
    for t in threads:
        t.join()
    assert [x.status_code for x in out] == [200, 200]
    m = c.get("/metrics").json()
    assert m["by_status"] == {"200": 2, "503": 1} and m["errors"] == {"overloaded": 1}


def test_large_body_is_parsed_off_the_event_loop(client, monkeypatch):
    where = []
    real = S.parse_body

    def spy(raw):
        try:
            asyncio.get_running_loop()
            where.append("loop")
        except RuntimeError:
            where.append("thread")
        return real(raw)

    monkeypatch.setattr(S, "parse_body", spy)
    assert post(client, ROW).status_code == 200
    assert post(client, _sized_body(PARSE_INLINE * 4)).status_code == 200
    assert post(client, b"[" * 40_000 + b"]" * 40_000).status_code == 400
    assert where == ["loop", "thread", "thread"]


def test_unknown_route_and_method(client):
    r = client.get("/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = client.get("/triage")
    assert r.status_code == 405 and r.json()["error"]["code"] == "method_not_allowed"


# ------------------------------------------------------------------ request ids

def test_request_id_echoed(client):
    r = post(client, ROW, headers={"X-Request-ID": "abc-123.x:y_z"})
    assert r.headers["x-request-id"] == "abc-123.x:y_z"
    assert client.triage.request_ids == ["abc-123.x:y_z"]


@pytest.mark.parametrize("given", [None, "", "has space", "x" * 129, "semi;colon", "café"])
def test_request_id_generated(client, given):
    headers = {} if given is None else {"X-Request-ID": given.encode("utf-8")}
    r = post(client, ROW, headers=headers)
    rid = r.headers["x-request-id"]
    assert len(rid) == 32 and int(rid, 16) >= 0 and rid != given


def test_request_id_on_errors(client):
    r = post(client, b"{", headers={"X-Request-ID": "err-1"})
    assert r.status_code == 400 and r.headers["x-request-id"] == "err-1"


def test_request_id_rule():
    assert request_id("A" * 128) == "A" * 128
    assert request_id("A" * 129) != "A" * 129
    assert request_id("a\nb") != "a\nb"


# ------------------------------------------------------------------ idempotency

def test_idempotent_replay(client):
    a = post(client, ROW, headers={"X-Request-ID": "first"})
    same = json.dumps(dict(reversed(list(ROW.items()))), indent=2)      # other key order and spacing
    b = post(client, same, headers={"X-Request-ID": "second"})
    assert a.status_code == b.status_code == 200
    assert a.content == b.content
    assert "x-idempotent-replay" not in a.headers and b.headers["x-idempotent-replay"] == "true"
    assert b.headers["x-request-id"] == "second"
    assert client.triage.calls == 1
    post(client, {**ROW, "turn_index": 4})
    assert client.triage.calls == 2


def test_concurrent_identical_requests_call_handle_once(make_client):
    c = make_client(FakeTriage(delay=0.3))
    out = []

    def go(i):
        out.append(post(c, ROW, headers={"X-Request-ID": f"t{i}"}))

    threads = [threading.Thread(target=go, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert c.triage.calls == 1
    assert all(r.status_code == 200 for r in out) and len({r.content for r in out}) == 1
    assert sum(r.headers.get("x-idempotent-replay") == "true" for r in out) == 7
    assert c.get("/metrics").json()["idempotency"]["in_flight"] == 0


def test_cache_is_lru(make_client):
    c = make_client(idem_size=2)
    rows = [{**ROW, "id": f"r{i}"} for i in range(3)]
    for row in rows:
        post(c, row)
    post(c, rows[2])
    post(c, rows[1])
    assert c.triage.calls == 3
    post(c, rows[0])                                  # evicted
    assert c.triage.calls == 4


def test_cache_off(make_client):
    c = make_client(idem_size=0)
    post(c, ROW)
    post(c, ROW)
    assert c.triage.calls == 2


def test_policy_change_clears_cache(client):
    """The very next request for the same body after a reload is recomputed, with or without a new version."""
    first = post(client, ROW)
    assert first.json()["confidence"] == 0.82
    client.triage.store.reload("test.2")
    r = post(client, ROW)
    assert client.triage.calls == 2 and "x-idempotent-replay" not in r.headers
    assert client.get("/metrics").json()["policy_version"] == "test.2"
    client.triage.store.reload()                                        # same version, edited rule or threshold
    client.triage.conf[id(client.triage.store.policy)] = 0.4
    r = post(client, ROW)
    assert client.triage.calls == 3 and "x-idempotent-replay" not in r.headers and r.json()["confidence"] == 0.4
    assert post(client, ROW).headers["x-idempotent-replay"] == "true"


def test_answer_made_under_an_old_policy_is_not_cached(make_client):
    """A request that started before a reload and ends after it is answered, but not cached."""
    fake = FakeTriage(delay=0.3)
    c = make_client(fake)
    t = threading.Thread(target=post, args=(c, ROW))
    t.start()
    deadline = time.monotonic() + 5
    while fake.calls == 0:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    fake.store.reload("test.2")
    post(c, {**ROW, "id": "new"})                                       # starts the new generation
    t.join()
    fake.delay = 0
    r = post(c, ROW)
    assert "x-idempotent-replay" not in r.headers and fake.calls == 3


def test_degraded_llm_answers_expire(make_client):
    """A CPU fallback made while the LLM step failed is cached for degraded_ttl_s only."""
    routed = {**ROW, "context": [{"speaker": "customer", "text": "route me"}]}
    c = make_client(FakeTriage(llm_status="breaker_open"), degraded_ttl_s=0.2)
    post(c, routed)
    assert post(c, routed).headers.get("x-idempotent-replay") == "true"
    time.sleep(0.25)
    assert "x-idempotent-replay" not in post(c, routed).headers and c.triage.calls == 2
    ok = make_client(FakeTriage(llm_status="ok"), degraded_ttl_s=0.2)
    post(ok, routed)
    time.sleep(0.25)
    assert post(ok, routed).headers.get("x-idempotent-replay") == "true"
    assert set(S.DEGRADED) == {"timeout", "http_error", "invalid", "breaker_open", "busy"}


def test_bad_llm_cost_is_not_counted(make_client):
    for cost in (-5e301, -1.0, 1e23, float("nan")):
        fake = FakeTriage()
        c = make_client(fake)
        orig = fake.handle

        def handle(obj, rid, orig=orig, cost=cost):
            resp, trace = orig(obj, rid)
            return resp, {**trace, "llm": {"status": "ok", "cost_usd": cost}}

        fake.handle = handle
        post(c, ROW)
        assert c.get("/metrics").json()["llm"]["cost_usd"] == 0


def test_policy_reload_without_version_change_reaches_cached_body(tmp_path, triage_shipped, split, monkeypatch):
    """Real Triage and PolicyStore: raise the thresholds without a new version; the same body routes at once."""
    from triage.policy import DEFAULT_POLICY, PolicyStore
    path = tmp_path / "policy.toml"
    shutil.copy(DEFAULT_POLICY, path)
    monkeypatch.setattr(triage_shipped, "store", PolicyStore(path, check_every_s=0.0))
    rows = [project(r) for r in split("dev")[:200]]
    row = next(r for r, (resp, tr) in zip(rows, triage_shipped.handle_many(rows))
               if not resp["needs_human"] and resp["confidence"] < 0.99)
    with TestClient(create_app(triage_shipped)) as c:
        assert post(c, row).json()["needs_human"] is False
        assert post(c, row).headers.get("x-idempotent-replay") == "true"
        text = path.read_text()
        path.write_text(re.sub(r"(?m)^(high|low) = [0-9.]+$", r"\1 = 0.99", text))
        assert path.read_text() != text and triage_shipped.store.current().thresholds == {"high": 0.99, "low": 0.99}
        r = post(c, row)
        assert "x-idempotent-replay" not in r.headers and r.json()["needs_human"] is True


# ------------------------------------------------------------------ failures inside handle

def test_handle_raising_gives_abstention(make_client):
    c = make_client(FakeTriage(fail=True))
    r = post(c, ROW)
    body = r.json()
    assert r.status_code == 200 and schema_valid(body)
    assert body["id"] == "row-1" and body["intent"] == "unknown" and body["action"] == "none"
    assert body["confidence"] == 0 and body["needs_human"] is True and body["error"]["code"] == "internal"
    assert MARKER not in r.text
    post(c, ROW)                                      # failures are not cached
    assert c.triage.calls == 2


def test_failure_caught_inside_handle_is_not_cached(make_client):
    class Caught(FakeTriage):
        def handle(self, obj, request_id):
            super().handle(obj, request_id)
            resp = {"id": obj["id"], "intent": "unknown", "action": "none", "confidence": 0.0, "needs_human": True,
                    "error": {"code": "internal", "message": "KeyError"}}
            return resp, {"flags": [], "route_reasons": ["internal_error"], "error": "KeyError"}

    c = make_client(Caught())
    assert post(c, ROW).json()["error"]["code"] == "internal"
    post(c, ROW)
    m = c.get("/metrics").json()
    assert c.triage.calls == 2 and m["errors"] == {"internal": 2} and m["abstentions"] == 2


@pytest.mark.parametrize("bad", [
    {"intent": "a", "action": "b", "confidence": float("nan"), "needs_human": False},
    {"intent": "a", "action": "b", "confidence": 1.5, "needs_human": False},
    {"intent": "a", "action": "b", "confidence": 0.5, "needs_human": "no"},
    {"intent": None, "action": "b", "confidence": 0.5, "needs_human": False},
    {"intent": "a", "action": "b", "confidence": 0.5, "needs_human": False, "cost_usd": -1},
])
def test_bad_handle_output_gives_abstention(make_client, bad):
    c = make_client(FakeTriage(response=bad))
    r = post(c, ROW)
    assert r.status_code == 200 and schema_valid(r.json()) and r.json()["error"]["code"] == "internal"


def test_unexpected_error_is_500_without_trace(make_client, monkeypatch):
    c = make_client()
    monkeypatch.setattr(c.app.state.service, "_answer", lambda *a: 1 / 0)
    r = post(c, ROW)
    assert r.status_code == 500 and r.json() == {"error": {"code": "server_error",
                                                           "message": "unexpected server error"}}
    assert "Traceback" not in r.text and "ZeroDivision" not in r.text


# ------------------------------------------------------------------ metrics and logs

def test_metrics_counters(client):
    post(client, ROW)
    post(client, ROW)                                                   # replay
    post(client, {**ROW, "id": "r2", "context": [{"speaker": "customer", "text": "route me"}]})
    post(client, b"{")
    post(client, b'{"context": []}')
    post(client, _sized_body(MAX_BODY + 1))
    m = client.get("/metrics").json()
    assert m["requests"] == 6
    assert m["by_status"] == {"200": 3, "400": 2, "413": 1}
    assert m["errors"] == {"bad_json": 1, "invalid_envelope": 1, "body_too_large": 1}
    assert m["replays"] == 1 and m["routed"] == 1 and m["abstentions"] == 0
    assert m["flags"] == {"empty_turns": 2, "second_intent": 1}
    assert m["route_reasons"] == {"low_confidence": 1}
    assert m["llm"] == {"calls": 1, "by_status": {"ok": 1}, "cost_usd": 0.0001}
    assert m["latency_ms"]["window"] == 6 and 0 <= m["latency_ms"]["p50"] <= m["latency_ms"]["p99"]
    assert m["idempotency"]["entries"] == 2 and m["policy_version"] == "test.1"


def test_percentiles():
    p = percentiles(range(1, 101))
    assert p == {"p50": 50.5, "p95": 95.05, "p99": 99.01}
    assert percentiles([]) == {"p50": None, "p95": None, "p99": None}


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(self.format(record))


def test_logs_have_no_customer_text(make_client):
    lines = _Lines()
    logger = logging.getLogger("triage.service")
    logger.addHandler(lines)
    try:
        c = make_client()
        row = {"id": f"id-{MARKER}", "context": [{"speaker": "customer", "text": f"route me {MARKER}"}]}
        post(c, row, headers={"X-Request-ID": "log-1", "X-Secret": MARKER, "Authorization": f"Bearer {MARKER}"})
        post(c, row, headers={"X-Request-ID": "log-2"})
        post(c, b'{"id": "' + MARKER.encode() + b'", "context": [', headers={"X-Request-ID": "log-3"})
        post(c, {"context": MARKER}, headers={"X-Request-ID": "log-4"})
        f = make_client(FakeTriage(fail=True))
        post(f, row, headers={"X-Request-ID": "log-5"})
    finally:
        logger.removeHandler(lines)
    assert len(lines.lines) == 5
    assert all(MARKER not in line for line in lines.lines)
    recs = [json.loads(line) for line in lines.lines]
    assert [r["request_id"] for r in recs] == ["log-1", "log-2", "log-3", "log-4", "log-5"]
    assert [r["status"] for r in recs] == [200, 200, 400, 400, 200]
    first = recs[0]
    assert {"ts", "latency_ms", "flags", "route_reasons", "llm_status", "llm_latency_ms", "llm_cost_usd",
            "policy_version", "needs_human", "abstained"} <= set(first)
    assert first["needs_human"] is True and first["llm_status"] == "ok" and first["llm_latency_ms"] == 250.0
    assert recs[1]["replay"] is True and recs[1]["llm_status"] is None
    assert recs[4]["error"] == "internal" and recs[4]["error_type"] == "RuntimeError"


# ------------------------------------------------------------------ the real app under uvicorn

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_uvicorn_end_to_end(tmp_path, split):
    if not (ROOT / "triage" / "pipeline.py").exists() or not BUNDLE.exists():
        pytest.skip("needs triage/pipeline.py and models/classical_v0.joblib")
    port = _free_port()
    env = {k: v for k, v in os.environ.items() if "API_KEY" not in k}
    env["TRIAGE_LLM"] = "off"
    log_path = tmp_path / "server.log"
    run = re.search(r"Run: `uvicorn (.+?)`", (ROOT / "docs" / "DESIGN.md").read_text()).group(1).split()
    run = [a.replace("0.0.0.0", "127.0.0.1").replace("$PORT", str(port)) for a in run]
    cmd = [sys.executable, "-m", "uvicorn", *run]                      # the run command docs/DESIGN.md gives
    with open(log_path, "wb") as out:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=out, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    rows = [project(r) for r in split("dev")[:5] + split("noisy")[:2] + split("hostile")]
    try:
        with httpx.Client(base_url=url, timeout=30) as h:
            deadline = time.monotonic() + 120
            while True:
                assert proc.poll() is None, log_path.read_text()[-2000:]
                try:
                    if h.get("/healthz").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                assert time.monotonic() < deadline, "service did not become ready"
                time.sleep(0.2)
            health = h.get("/healthz").json()
            assert health["status"] == "ok" and health["uptime_s"] >= 0
            for row in rows:
                r = h.post("/triage", json=row, headers={"X-Request-ID": "e2e-" + row["id"]})
                assert r.status_code == 200 and r.headers["x-request-id"] == "e2e-" + row["id"]
                assert schema_valid(r.json()) and r.json()["id"] == row["id"]
            assert h.post("/triage", content=_sized_body(MAX_BODY + 1)).status_code == 413
            assert h.post("/triage", content=b"{").status_code == 400
            m = h.get("/metrics").json()
            assert m["by_status"].get("200") == len(rows) and m["llm"]["calls"] == 0
    finally:
        proc.terminate()
        proc.wait(15)
    text = log_path.read_text()
    recs = [json.loads(line) for line in text.splitlines()]           # stdout holds JSON lines only
    assert sum("request_id" in r for r in recs) == len(rows) + 2
    for row in rows:
        for t in row["context"]:
            if len(t["text"]) >= 20 and t["speaker"] != "action":
                assert t["text"] not in text


# ------------------------------------------------------------------ scripts/loadtest.py against the fake

def test_loadtest_script(tmp_path):
    import uvicorn
    spec = importlib.util.spec_from_file_location("loadtest", ROOT / "scripts" / "loadtest.py")
    lt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lt)
    rows = [{**ROW, "id": f"lt-{i}", "intent": "gold-only"} for i in range(6)]
    inputs = tmp_path / "rows.jsonl"
    inputs.write_text("".join(json.dumps(r) + "\n" for r in rows))
    fake, port = FakeTriage(), _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(fake), host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    try:
        out, preds = tmp_path / "bench.json", tmp_path / "preds.jsonl"
        code = lt.main(["--url", f"http://127.0.0.1:{port}", "--inputs", str(inputs), "--limit", "5", "--warmup", "2",
                        "--repeat", "2", "--concurrency", "3", "--out", str(out), "--pred-out", str(preds),
                        "--label", "test", "--wait-s", "10"])
    finally:
        server.should_exit = True
        th.join(10)
    rep = json.loads(out.read_text())
    assert code == 0 and rep["label"] == "test" and rep["sample_count"] == 10
    assert rep["errors"] == rep["non_200"] == rep["schema_invalid"] == rep["id_mismatch"] == 0
    assert rep["replays"] == 5 and fake.calls == 7                     # 2 warm-up + 5 rows; the second pass replays
    assert {"p50", "p95", "p99", "max", "mean"} == set(rep["latency_ms"]) and rep["throughput_rps"] > 0
    assert rep["server_health"]["http_status"] == 200
    assert rep["server_metrics_after"]["requests"] - rep["server_metrics_before"]["requests"] == 10
    assert rep["llm_calls"] == 0 and rep["llm_cost_usd"] == 0 and rep["cost_per_message_usd"] == 0
    recs = rep["requests"]
    assert [r["seq"] for r in recs] == list(range(10)) and len({r["request_id"] for r in recs}) == 10
    assert all(r["status"] == 200 and r["schema_valid"] and r["id_echo"] for r in recs)
    lines = [json.loads(line) for line in preds.read_text().splitlines()]
    assert [p["id"] for p in lines] == [f"lt-{i}" for i in range(5)] and all(schema_valid(p) for p in lines)
    assert all(r.startswith("lt-") for r in fake.request_ids)
    assert fake.keys == {"id", "turn_index", "context"}                # gold-only fields never sent
