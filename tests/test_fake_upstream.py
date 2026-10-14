"""The LLM step against experiments/chaos/fake_upstream.py in its "strict" mode (a gateway that knows only the OpenAI
Chat Completions fields the openai dialect sends) and its "ratelimit" mode (429 with Retry-After). All on 127.0.0.1;
no API key, no real provider."""
from __future__ import annotations

import json
import logging

import httpx
import pytest
from conftest import fake_upstream_module
from fastapi.testclient import TestClient

from triage.llm_action import OPENAI_FIELDS, LLMActionStep, LLMConfig, request_body
from triage.service import create_app

KEY = "test-key"
ROW = {"id": "8645-22", "turn_index": 22, "context": [
    {"speaker": "agent", "text": "can i have your zip code, email address, or phone number?"},
    {"speaker": "customer", "text": "<phone>"},
    {"speaker": "action", "text": "account has been pulled up for rodriguez domingo."},
    {"speaker": "customer", "text": "oh"}]}
OPENROUTER_ONLY = ("max_tokens", "provider", "reasoning", "usage")


class StubRetriever:
    def examples(self, row, k=8):
        return []


def step_for(url: str, **kw) -> LLMActionStep:
    return LLMActionStep(LLMConfig(mode="on", base_url=url, api_key=KEY, **kw), StubRetriever())


def test_fake_knows_the_fields_the_openai_dialect_sends():
    """The fake's allowed set is the set the service sends in the openai dialect, nothing more."""
    fu = fake_upstream_module()
    assert fu.STRICT_FIELDS == OPENAI_FIELDS
    body = request_body(LLMConfig(base_url="http://127.0.0.1:1/v1"), "S", "U")
    assert set(body) == OPENAI_FIELDS and fu.unknown_fields(body) == []
    old = request_body(LLMConfig(), "S", "U")                          # the OpenRouter body, unchanged
    assert fu.unknown_fields(old) == sorted(OPENROUTER_ONLY)
    assert fu.unknown_fields({**body, "messages": [{"role": "user", "content": "U", "name": "x"}],
                              "response_format": {**body["response_format"], "extra": 1}}) == \
        ["messages[0].name", "response_format.extra"]
    assert {"strict", "ratelimit"} <= set(fu.MODES)


def test_strict_accepts_the_openai_dialect(fake_upstream):
    url, state = fake_upstream("strict")
    step = step_for(url)                                               # 127.0.0.1: auto picks openai
    assert step.cfg.dialect == "openai"
    res = step.decide(ROW)
    assert (res.status, res.http_status, res.usable) == ("ok", 200, True)
    assert res.action == "pull-up-account"                             # the fake's fixed answer
    assert res.model == step.cfg.model and res.provider == "fake-upstream"
    assert res.response_id and res.response_id.startswith("fake-")
    assert state.calls["outcome:strict"] == 1 and not state.calls["outcome:strict_rejected"]
    step.close()


def test_strict_rejects_the_openrouter_body(fake_upstream):
    """Told to use OpenRouter's names, every call fails with 400 and the breaker opens after five: the LLM step goes
    quiet with no error to the caller. This is the failure the openai dialect avoids."""
    url, state = fake_upstream("strict")
    step = step_for(url, dialect_setting="openrouter")
    assert step.cfg.dialect == "openrouter"
    for _ in range(5):
        res = step.decide(ROW)
        assert (res.status, res.http_status, res.usable) == ("http_error", 400, False)
        assert all(f in res.error for f in OPENROUTER_ONLY), res.error
        assert KEY not in res.error
    assert step.breaker.state == "open" and step.decide(ROW).status == "breaker_open"
    assert state.calls["outcome:strict_rejected"] == 5 and not state.calls["outcome:strict"]
    step.close()


def test_strict_rejects_the_old_plain_body(fake_upstream):
    """Before the openai dialect, a non-OpenRouter URL got "reasoning" and "max_tokens"; the strict fake says 400."""
    url, _ = fake_upstream("strict")
    old = {**request_body(LLMConfig(base_url=url), "S", "U")}
    for k in ("max_completion_tokens", "reasoning_effort"):
        old.pop(k)
    old.update(max_tokens=2048, reasoning={"effort": "none"})
    r = httpx.post(url + "/chat/completions", json=old, timeout=5)
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error" and err["param"] == "max_tokens"
    assert "max_tokens" in err["message"] and "reasoning" in err["message"]


def test_ratelimit_is_an_http_error(fake_upstream):
    url, state = fake_upstream("ratelimit")
    r = httpx.post(url + "/chat/completions", json=request_body(LLMConfig(base_url=url), "S", "U"), timeout=5)
    assert r.status_code == 429 and r.headers["retry-after"] == "1"
    assert r.json()["error"]["code"] == "rate_limit_exceeded"
    step = step_for(url)
    res = step.decide(ROW)
    assert (res.status, res.http_status, res.usable) == ("http_error", 429, False)
    assert res.cost_usd is None and res.response_id is None
    assert state.calls["outcome:ratelimit"] == 2
    step.close()


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def test_strict_gateway_through_the_app_logs_provenance(fake_upstream, triage_off, monkeypatch):
    """Strict fake -> real LLMActionStep (auto picks openai for 127.0.0.1) -> real Triage -> POST /triage: the call
    succeeds, and the returned model, provider and response id reach the trace, /healthz and the log line, which
    holds none of the conversation's text."""
    url, state = fake_upstream("strict")
    step = step_for(url, gate=1.0)                                     # gate 1.0: an unflagged row always asks
    monkeypatch.setattr(triage_off, "llm", step)
    lines, logger = _Lines(), logging.getLogger("triage.service")
    try:
        with TestClient(create_app(triage_off)) as c:
            logger.addHandler(lines)
            r = c.post("/triage", json=ROW, headers={"X-Request-ID": "strict-1"})
            health = c.get("/healthz").json()["llm"]
    finally:
        logger.removeHandler(lines)
        step.close()
    assert r.status_code == 200 and r.json()["action"] == "pull-up-account"
    assert (health["dialect"], health["dialect_setting"]) == ("openai", "auto")
    assert state.calls["outcome:strict"] == 1 and not state.calls["outcome:strict_rejected"]
    (line,) = [x for x in lines.lines if '"request_id":"strict-1"' in x]
    rec = json.loads(line)
    assert (rec["llm_status"], rec["llm_model"], rec["llm_provider"]) == ("ok", step.cfg.model, "fake-upstream")
    assert rec["llm_response_id"].startswith("fake-")
    assert not [t["text"] for t in ROW["context"] if len(t["text"]) >= 4 and t["text"] in line]


@pytest.mark.parametrize("mode", ["strict", "ratelimit"])
def test_new_modes_can_be_switched_on(fake_upstream, mode):
    url, state = fake_upstream("ok")
    base = url.rsplit("/v1", 1)[0]
    assert httpx.post(f"{base}/_mode?mode={mode}", timeout=5).json() == {"mode": mode}
    assert httpx.get(f"{base}/_state", timeout=5).json()["mode"] == mode
