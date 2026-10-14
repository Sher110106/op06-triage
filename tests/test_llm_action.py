"""The LLM action step against a fake OpenAI-compatible server on 127.0.0.1 (no real API calls), plus the
retrieval index check against the examples the LLM test used."""
from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from triage import prompts as P
from triage.data import ROOT, project
from triage.llm_action import (CircuitBreaker, LLMActionStep, LLMConfig, LLMResult, list_cost, list_cost_uncached,
                                request_body)
from triage.retrieval import DEFAULT_PATH, Retriever

KEY = "sk-or-v1-TESTKEY-0123456789abcdef"
ROW = {"id": "8645-22", "turn_index": 22, "context": [
    {"speaker": "agent", "text": "can i have your zip code, email address, or phone number?"},
    {"speaker": "customer", "text": "<phone>"},
    {"speaker": "action", "text": "account has been pulled up for rodriguez domingo."},
    {"speaker": "customer", "text": "oh"}]}
GOOD = {"intent": "recover_password", "action": "enter-details", "needs_human": False, "confidence": 0.9}
USAGE = {"prompt_tokens": 10000, "completion_tokens": 40, "cost": 0.000123,
         "prompt_tokens_details": {"cached_tokens": 8000, "cache_write_tokens": 0}}


def completion(content, usage=USAGE) -> dict:
    return {"id": "gen-1", "model": "openai/gpt-6-luna-20260922", "provider": "OpenAI",
            "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": usage}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        srv = self.server
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        with srv.lock:
            srv.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
        mode = srv.mode
        code, out = 200, None
        if mode == "valid":
            out = completion(json.dumps(GOOD))
        elif mode == "delay":
            time.sleep(0.2)
            out = completion(json.dumps(GOOD))
        elif mode == "prose":
            out = completion("The customer wants a new password, so the agent should enter details.")
        elif mode == "unknown_action":
            out = completion(json.dumps(dict(GOOD, action="refund-everything")))
        elif mode == "abstain":
            out = completion(json.dumps({"intent": "unknown", "action": "none", "needs_human": True, "confidence": 0}))
        elif mode == "none_content":
            out = completion(None)
        elif mode in ("500", "429"):
            code, out = int(mode), {"error": {"message": f"upstream said {mode}", "code": int(mode)}}
        elif mode == "error_body":
            out = {"error": {"message": "provider returned error", "code": 502}}
        elif mode == "echo_key":
            code, out = 401, {"error": {"message": "bad header: " + self.headers.get("Authorization", "")}}
        elif mode == "echo_key_long":                                  # the key straddles the 200-char cut
            code, out = 401, "x" * 179 + self.headers.get("Authorization", "")
        elif mode == "html":
            code, out = 502, "<html>bad gateway</html>"
        elif mode == "slow":
            time.sleep(1.0)
            out = completion(json.dumps(GOOD))
        elif mode == "brace_flood":                                    # 50 KB of "{" as the answer
            out = completion("{" * 50_000)
        elif mode == "long_answer":                                    # under the reply cap, over the content cap
            out = completion(json.dumps(GOOD) + " " * 20_000)
        elif mode == "huge_reply":
            out = completion(json.dumps(GOOD), usage=dict(USAGE, padding="x" * 300_000))
        elif mode.startswith("usage:"):
            out = completion(json.dumps(GOOD), usage=json.loads(mode[6:]))
        data = (json.dumps(out) if isinstance(out, dict) else out).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            pass                                                   # client gave up (timeout test)


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.daemon_threads = True
    srv.lock, srv.requests, srv.mode = threading.Lock(), [], "valid"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def fake(server):
    server.mode = "valid"
    with server.lock:
        server.requests.clear()
    return server


class StubRetriever:
    def examples(self, row, k=8):
        return []

    def top_k(self, row, k=8):
        return []


def make_step(srv, path="/v1", **kw) -> LLMActionStep:
    port = srv.server_address[1]
    cfg = LLMConfig(mode="on", base_url=f"http://127.0.0.1:{port}{path}", api_key=KEY, **kw)
    return LLMActionStep(cfg, StubRetriever())


def no_key(*objs):
    """Neither the key nor any 10-character piece of it."""
    for o in objs:
        for text in (repr(o), str(o)):
            assert not any(KEY[i:i + 10] in text for i in range(len(KEY) - 9))


# ----------------------------------------------------------------------------- config

def test_config_from_env():
    c = LLMConfig.from_env({})
    assert (c.mode, c.enabled, c.base_url, c.model, c.provider, c.timeout_s, c.gate) == (
        "auto", False, "https://openrouter.ai/api/v1", "openai/gpt-6-luna-20260922", "openai", 4.0, 0.642841)
    assert c.openrouter
    assert LLMConfig.from_env({"OPENROUTER_API_KEY": KEY}).enabled
    assert LLMConfig.from_env({"OPENROUTER_API_KEY": "a", "TRIAGE_LLM_API_KEY": KEY}).api_key == KEY
    assert not LLMConfig.from_env({"TRIAGE_LLM": "off", "OPENROUTER_API_KEY": KEY}).enabled
    assert LLMConfig.from_env({"TRIAGE_LLM": "on"}).enabled
    c = LLMConfig.from_env({"TRIAGE_LLM_BASE_URL": "http://gw.local/v1/", "TRIAGE_LLM_TIMEOUT_S": "2.5",
                            "TRIAGE_LLM_GATE": "0.5", "TRIAGE_LLM_MODEL": "m", "OPENROUTER_API_KEY": KEY})
    assert (c.base_url, c.timeout_s, c.gate, c.model, c.openrouter) == ("http://gw.local/v1", 2.5, 0.5, "m", False)
    for env in ({"TRIAGE_LLM": "maybe"}, {"TRIAGE_LLM_TIMEOUT_S": "fast"}, {"TRIAGE_LLM_TIMEOUT_S": "0"},
                {"TRIAGE_LLM_GATE": "2"}):
        with pytest.raises(ValueError) as e:
            LLMConfig.from_env(dict(env, OPENROUTER_API_KEY=KEY))
        assert KEY not in str(e.value)


def test_key_hidden_in_config():
    c = LLMConfig.from_env({"OPENROUTER_API_KEY": KEY, "TRIAGE_LLM_BASE_URL": "https://u:p@gw.local/v1?token=t"})
    no_key(c, c.public())
    assert c.public()["key_set"] and c.public()["base_url"] == "https://gw.local/v1"


# ----------------------------------------------------------------------------- request

def test_request_body_fields():
    plain = request_body(LLMConfig(base_url="http://gw.local/v1"), "S", "U")
    assert plain["model"] == "openai/gpt-6-luna-20260922"
    assert plain["messages"] == [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    assert plain["response_format"] == {"type": "json_schema", "json_schema": {
        "name": "triage", "strict": True, "schema": P.output_schema()}}
    assert plain["max_tokens"] == 2048 and plain["reasoning"] == {"effort": "none"}
    assert "temperature" not in plain and "provider" not in plain and "usage" not in plain
    orr = request_body(LLMConfig(provider="openai"), "S", "U")
    assert orr["provider"] == {"order": ["openai"], "allow_fallbacks": False, "require_parameters": True}
    assert orr["usage"] == {"include": True}


def test_valid_answer_and_request(fake):
    step = make_step(fake)
    res = step.decide(ROW)
    assert res.status == "ok" and res.usable
    assert (res.intent, res.action, res.confidence, res.needs_human) == (
        "recover_password", "enter-details", 0.9, False)
    assert (res.prompt_tokens, res.completion_tokens, res.cached_tokens, res.cache_write_tokens) == (10000, 40, 8000, 0)
    assert res.cost_usd == pytest.approx((2000 * 0.10 + 8000 * 0.01 + 40 * 0.50) / 1e6)
    assert res.cost_uncached_usd == pytest.approx((10000 * 0.10 + 40 * 0.50) / 1e6)      # no credit for the cache
    assert res.cost_uncached_usd > res.cost_usd
    assert res.cost_reported == 0.000123
    assert (res.model, res.provider, res.http_status) == ("openai/gpt-6-luna-20260922", "OpenAI", 200)
    assert res.latency_s > 0
    (req,) = fake.requests
    assert req["path"] == "/v1/chat/completions"
    assert req["headers"]["Authorization"] == f"Bearer {KEY}"
    b = req["body"]
    assert b["messages"][0] == {"role": "system", "content": P.static_prompt()}
    user = b["messages"][1]["content"]
    nonce = user.split("<<<CONVERSATION ")[1].split(">>>")[0]
    assert user == P.dynamic_prompt(ROW, [], nonce)[0]
    assert "provider" not in b and "usage" not in b and b["reasoning"] == {"effort": "none"}
    no_key(res, step.health())
    step.close()


def test_openrouter_url_sends_provider_block(fake):
    step = make_step(fake, path="/openrouter.ai/api/v1", provider="openai")
    assert step.decide(ROW).status == "ok"
    (req,) = fake.requests
    assert req["path"] == "/openrouter.ai/api/v1/chat/completions"
    assert req["body"]["provider"] == {"order": ["openai"], "allow_fallbacks": False, "require_parameters": True}
    assert req["body"]["usage"] == {"include": True}
    step.close()


def test_no_auth_header_without_key(fake):
    port = fake.server_address[1]
    step = LLMActionStep(LLMConfig(mode="on", base_url=f"http://127.0.0.1:{port}/v1"), StubRetriever())
    assert step.decide(ROW).status == "ok"
    assert "Authorization" not in fake.requests[0]["headers"]
    step.close()


@pytest.mark.parametrize("mode, status, code", [
    ("prose", "invalid", 200), ("unknown_action", "invalid", 200), ("none_content", "invalid", 200),
    ("abstain", "abstained", 200), ("500", "http_error", 500), ("429", "http_error", 429),
    ("error_body", "http_error", 200), ("echo_key", "http_error", 401), ("echo_key_long", "http_error", 401),
    ("html", "http_error", 502),
])
def test_bad_answers(fake, mode, status, code):
    fake.mode = mode
    step = make_step(fake)
    res = step.decide(ROW)
    assert (res.status, res.http_status, res.usable) == (status, code, False)
    if status == "invalid":
        assert res.intent is None and res.action is None and res.error
    if status == "abstained":
        assert (res.intent, res.action) == ("unknown", "none")
    no_key(res, res.as_dict())
    step.close()


def test_timeout_is_a_hard_deadline(fake):
    fake.mode = "slow"                                                 # answers after 1.0 s
    step = make_step(fake, timeout_s=0.3)
    t0 = time.perf_counter()
    res = step.decide(ROW)
    wall = time.perf_counter() - t0
    assert res.status == "timeout" and not res.usable
    assert wall < 0.6 and res.latency_s < 0.6
    no_key(res)
    step.close()


def test_connection_refused():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    step = LLMActionStep(LLMConfig(mode="on", base_url=f"http://127.0.0.1:{port}/v1", api_key=KEY), StubRetriever())
    res = step.decide(ROW)
    assert res.status == "http_error" and res.http_status is None and res.error == "ConnectError"
    no_key(res)
    step.close()


def test_disabled_makes_no_call(fake):
    port = fake.server_address[1]
    step = LLMActionStep(LLMConfig(mode="auto", base_url=f"http://127.0.0.1:{port}/v1"), StubRetriever())
    assert step.decide(ROW) == LLMResult("disabled")
    assert fake.requests == []
    step.close()


def test_internal_error_never_raises(fake):
    class Broken:
        def examples(self, row, k=8):
            raise RuntimeError(f"boom {KEY}")

    step = LLMActionStep(make_step(fake).cfg, Broken())
    res = step.decide(ROW)
    assert res.status == "invalid" and res.error == "internal: RuntimeError"
    no_key(res)
    step.close()
    step = make_step(fake)
    assert step.decide({"id": "bad"}).status == "invalid"              # no context at all
    step.close()


# ----------------------------------------------------------------------------- breaker

def test_breaker_opens_and_recovers(fake):
    fake.mode = "500"
    step = make_step(fake, breaker_open_s=0.3)
    assert [step.decide(ROW).status for _ in range(5)] == ["http_error"] * 5
    assert step.breaker.state == "open"
    assert step.decide(ROW).status == "breaker_open" and len(fake.requests) == 5
    time.sleep(0.35)
    assert step.decide(ROW).status == "http_error"                     # failed trial opens it again
    assert step.decide(ROW).status == "breaker_open" and len(fake.requests) == 6
    time.sleep(0.35)
    fake.mode = "valid"
    assert step.breaker.state == "half_open"
    assert step.decide(ROW).status == "ok"                             # good trial closes it
    assert step.breaker.state == "closed"
    assert step.decide(ROW).status == "ok"
    step.close()


def test_breaker_counts_timeouts_and_invalid_but_not_abstained():
    b = CircuitBreaker(failures=3, open_s=10.0)
    for ok in (False, False, True, False, False):                     # a success resets the streak
        assert b.allow()
        b.record(ok)
    assert b.state == "closed"
    b.record(False)
    assert b.state == "open" and not b.allow()


def test_half_open_lets_exactly_one_call_through(fake):
    fake.mode = "500"
    step = make_step(fake, breaker_open_s=0.2)
    for _ in range(5):
        step.decide(ROW)
    time.sleep(0.25)
    fake.mode = "delay"                                                # trial takes 0.2 s
    out, start = [], threading.Barrier(8)

    def go():
        start.wait()
        out.append(step.decide(ROW).status)

    ts = [threading.Thread(target=go) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sorted(out) == ["breaker_open"] * 7 + ["ok"]
    assert step.breaker.state == "closed"
    step.close()


def test_eight_threads_at_once(fake):
    fake.mode = "valid"
    step = make_step(fake)
    out, lock = [], threading.Lock()

    def go():
        for _ in range(10):
            r = step.decide(ROW)
            with lock:
                out.append(r.status)

    ts = [threading.Thread(target=go) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert out == ["ok"] * 80 and len(fake.requests) == 80
    fake.mode = "500"
    out.clear()
    ts = [threading.Thread(target=go) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert set(out) == {"http_error", "breaker_open"}
    assert 5 <= out.count("http_error") <= 5 + 7                      # at most the calls already in flight
    step.close()


@pytest.mark.parametrize("mode, error", [("brace_flood", "answer over"), ("long_answer", "answer over"),
                                         ("huge_reply", "reply over")])
def test_oversized_answers_are_invalid_and_fast(fake, mode, error):
    """A broken gateway's huge reply cannot freeze the process: the reply and the parsed text are both capped."""
    fake.mode = mode
    step = make_step(fake)
    t0 = time.perf_counter()
    res = step.decide(ROW)
    assert res.status == "invalid" and res.error.startswith(error) and not res.usable
    assert time.perf_counter() - t0 < 1.0
    step.close()


@pytest.mark.parametrize("usage", [
    {"prompt_tokens": "lots", "completion_tokens": -1e308},
    {"prompt_tokens": 100, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 1e9}},
    {"prompt_tokens": 1e30, "completion_tokens": 10},
    {"prompt_tokens": -5, "completion_tokens": 10},
])
def test_bad_token_counts_give_no_cost(fake, usage):
    fake.mode = "usage:" + json.dumps(usage)
    step = make_step(fake)
    res = step.decide(ROW)
    assert res.status == "ok" and res.cost_usd is None and res.cost_uncached_usd is None
    assert all(0 <= n <= 10_000_000 for n in (res.prompt_tokens, res.completion_tokens, res.cached_tokens,
                                              res.cache_write_tokens))
    fake.mode = "usage:" + json.dumps({"prompt_tokens": 100, "completion_tokens": 10, "cost": -3})
    res = step.decide(ROW)
    assert res.cost_usd == pytest.approx(1.5e-5) and res.cost_reported is None
    assert res.cost_uncached_usd == pytest.approx(1.5e-5)                  # nothing cached: the two costs agree
    step.close()


def test_busy_when_max_in_flight_calls_wait(fake):
    """A slow provider holds at most max_in_flight threads; the next call comes back "busy" at once."""
    fake.mode = "slow"                                                 # answers after 1.0 s
    step = make_step(fake, max_in_flight=2)
    out = []
    ts = [threading.Thread(target=lambda: out.append(step.decide(ROW).status)) for _ in range(2)]
    for t in ts:
        t.start()
    deadline = time.monotonic() + 5
    while len(fake.requests) < 2:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    t0 = time.perf_counter()
    assert step.decide(ROW) == LLMResult("busy")
    assert time.perf_counter() - t0 < 0.1
    for t in ts:
        t.join()
    assert out == ["ok", "ok"] and step.breaker.state == "closed"
    assert step.decide(ROW).status == "ok"                             # slots come back
    step.close()


def test_list_cost():
    assert list_cost("openai/gpt-6-luna-20260922", 1_000_000, 1_000_000, 0, 0) == pytest.approx(0.60)
    assert list_cost("openai/gpt-6-luna-20260922", 1_000_000, 0, 500_000, 500_000) == pytest.approx(
        0.005 + 0.0625)
    assert list_cost("other/model", 10, 10, 0, 0) is None


def test_list_cost_uncached():
    """MODELS.md prices a benchmark at uncached rates: every prompt token at the input price."""
    assert list_cost_uncached("openai/gpt-6-luna-20260922", 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert list_cost_uncached("openai/gpt-6-luna", 3000, 50) == pytest.approx((3000 * 0.10 + 50 * 0.50) / 1e6)
    assert list_cost_uncached("other/model", 10, 10) is None
    unc = list_cost_uncached("openai/gpt-6-luna", 3000, 50)
    assert unc == list_cost("openai/gpt-6-luna", 3000, 50, 0, 0)          # nothing cached: the same price
    assert unc > list_cost("openai/gpt-6-luna", 3000, 50, 2500, 0)        # cache reads are a discount
    # a cache write costs more than a plain input token (0.125 against 0.10), so it can put the cached-rate
    # price above the uncached one
    assert unc < list_cost("openai/gpt-6-luna", 3000, 50, 0, 2500)


# ----------------------------------------------------------------------------- retrieval

@pytest.fixture(scope="session")
def retriever():
    if not DEFAULT_PATH.exists():
        pytest.skip("models/retrieval.joblib missing: run scripts/build_retrieval.py")
    return Retriever.load(DEFAULT_PATH)


def test_retrieval_matches_llm_test_examples(retriever, split):
    ref = json.loads((ROOT / "experiments" / "llm_tail" / "retrieval.json").read_text())
    rows = [project(r) for r in split("dev") + split("noisy")]
    differ = []
    for r in rows:
        got = retriever.top_k(r)
        if [i for i, _ in got] != [i for i, _ in ref[r["id"]]]:
            # only the order of exactly tied candidates may differ: the tested examples used an unstable sort
            assert [round(s, 4) for _, s in got] == [s for _, s in ref[r["id"]]], r["id"]
            differ.append(r["id"])
    print(f"retrieval top-8 differs only by tie order on {len(differ)}/{len(rows)}: {differ}")
    assert len(differ) <= 4
    r = rows[0]
    got = retriever.top_k(r)
    assert [round(s, 4) for _, s in got] == [s for _, s in ref[r["id"]]]
    assert len({i.rsplit("-", 1)[0] for i, _ in got}) == 8
    ex = retriever.examples(r)
    assert [e["id"] for e in ex] == [i for i, _ in got]
    assert set(ex[0]) == {"id", "intent", "action", "context"}


def test_step_with_real_retriever(fake, retriever):
    port = fake.server_address[1]
    step = LLMActionStep(LLMConfig(mode="on", base_url=f"http://127.0.0.1:{port}/v1", api_key=KEY), retriever)
    assert step.decide(ROW).status == "ok"
    user = fake.requests[0]["body"]["messages"][1]["content"]
    nonce = user.split("<<<CONVERSATION ")[1].split(">>>")[0]
    assert user == P.dynamic_prompt(ROW, retriever.examples(ROW), nonce)[0]
    assert user.count("\nExample ") == 8
    step.close()
