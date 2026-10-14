"""A local stand-in for the OpenAI-compatible chat-completions provider, for fault injection. No API key, no network.

  .venv/bin/python experiments/chaos/fake_upstream.py --port 18001 --mode ok [--delay-s 1.5] [--replay FILE]

Serves POST <any path>/chat/completions on 127.0.0.1. The service's LLM step posts to $TRIAGE_LLM_BASE_URL +
"/chat/completions", so set TRIAGE_LLM_BASE_URL=http://127.0.0.1:<port>/v1.

Modes (what a call gets back):
  ok              after --delay-s (1.5 s, about Luna's p50): a valid chat-completion whose content is the recorded
                  GPT-6 Luna answer for this conversation (--replay; matched on the gate-cleaned turns and turn index,
                  the same text the service puts in the prompt), or, when no recorded answer matches, a fixed answer
                  (intent and action from the request's own enum lists: the first action named in FIXED_ACTIONS that
                  the enum allows)
  error500        at once: HTTP 500 with an OpenAI-style error object
  hang            sleeps 30 s, then tries to answer as "ok" (the caller has long given up)
  garbage         HTTP 200, Content-Type text/plain, an English sentence: the body is not JSON at all
  garbage_content HTTP 200, a valid chat-completion envelope whose message content is prose with no JSON object
  bad_enum        HTTP 200, a valid envelope whose content is JSON with an action outside the enum
  slow            3.5 s, then as "ok"
  flaky           each call: error500 with probability 0.5 (seeded), else as "ok"
  constant        after --delay-s: valid JSON, in the enum, but always the same answer (intent the first in the
                  request's enum, action "offer-refund"): a provider that is up and well-formed but wrong
  huge            at once: HTTP 200, a valid envelope whose content is 1,000,000 characters (a 1 MB body)
  strict          a gateway that knows only the OpenAI Chat Completions fields the service's openai dialect sends
                  (STRICT_FIELDS, plus role/content per message and type/json_schema/name/strict/schema in
                  response_format): any other field gets HTTP 400 at once, OpenAI-style, naming the fields; a body
                  with none answers as "ok". OpenRouter's names (max_tokens, reasoning, usage, provider) get the 400.
  ratelimit       at once: HTTP 429, Retry-After: --retry-after-s (1), an OpenAI-style rate_limit_exceeded error

Control and state (for the test driver):
  GET  /_state           JSON: mode, calls by outcome, replay hits and misses, calls in flight now and at most
  POST /_mode?mode=NAME  switch the mode while running (the recovery test)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODES = ("ok", "error500", "hang", "garbage", "garbage_content", "bad_enum", "slow", "flaky", "constant", "huge",
         "strict", "ratelimit")
# the fields of triage.llm_action.OPENAI_FIELDS, written out so this file stands alone (a test checks they match)
STRICT_FIELDS = frozenset({"model", "messages", "response_format", "max_completion_tokens", "reasoning_effort"})
_STRICT_MESSAGE = frozenset({"role", "content"})
_STRICT_FORMAT = frozenset({"type", "json_schema"})
_STRICT_SCHEMA = frozenset({"name", "strict", "schema"})
FIXED_ACTIONS = ("pull-up-account", "verify-identity", "search-faq")
HEAD = re.compile(r"^CONVERSATION TO LABEL: (\d+) turns, one JSON object per line"
                  r"(?:; the next action happens at turn (\d+) of the full conversation)?\.$")
START = re.compile(r"^<<<CONVERSATION ([0-9a-f]+)>>>$")


def unknown_fields(body: dict) -> list[str]:
    """Fields the strict mode does not know: top-level names outside STRICT_FIELDS, then names inside messages and
    response_format, as "messages[0].name" and "response_format.json_schema.x". Sorted within each level."""
    out = sorted(k for k in body if k not in STRICT_FIELDS)
    msgs = body.get("messages")
    for i, m in enumerate(msgs if isinstance(msgs, list) else []):
        if isinstance(m, dict):
            out += [f"messages[{i}].{k}" for k in sorted(m) if k not in _STRICT_MESSAGE]
    rf = body.get("response_format")
    if isinstance(rf, dict):
        out += [f"response_format.{k}" for k in sorted(rf) if k not in _STRICT_FORMAT]
        js = rf.get("json_schema")
        if isinstance(js, dict):
            out += [f"response_format.json_schema.{k}" for k in sorted(js) if k not in _STRICT_SCHEMA]
    return out


def conv_key(n_turns: int, turn_index, lines: list[str]) -> str:
    """The replay key: turn count, turn index and the prompt's JSON line per turn."""
    h = hashlib.sha256(f"{n_turns}|{turn_index}|".encode())
    h.update("\n".join(lines).encode("utf-8", "surrogatepass"))
    return h.hexdigest()


def key_from_prompt(user: str) -> str | None:
    """Rebuild the key from the user message the service sent (triage.prompts.dynamic_prompt)."""
    lines = user.split("\n")
    for i, line in enumerate(lines):
        m = HEAD.match(line)
        if not m or i + 1 >= len(lines):
            continue
        s = START.match(lines[i + 1])
        if not s:
            continue
        end = f"<<<END CONVERSATION {s.group(1)}>>>"
        try:
            j = lines.index(end, i + 2)
        except ValueError:
            return None
        n = int(m.group(1))
        ti = int(m.group(2)) if m.group(2) is not None else None
        return conv_key(n, ti, lines[i + 2:j])
    return None


def build_replay(path: Path) -> tuple[dict, dict]:
    """key -> recorded content text, for every dev + dev_noisy row with a recorded answer. Rows that the gate
    cleans to the same conversation share a key; when their recorded answers differ the key is ambiguous and the
    first row's answer is served (counted in /_state)."""
    from triage import gate
    from triage import prompts as P
    from triage.data import load_split, project

    rec = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if isinstance(r.get("text"), str):
            rec[r["id"]] = r["text"]
    table: dict[str, list[tuple[str, str]]] = {}
    for split in ("dev", "noisy"):
        for row in load_split(split):
            if row["id"] not in rec:
                continue
            g = gate.clean(project(row))
            if g.abstain_reason:
                continue
            k = conv_key(len(g.context), g.turn_index, [P._turn_json(t) for t in g.context])
            table.setdefault(k, []).append((row["id"], rec[row["id"]]))
    out = {k: v[0][1] for k, v in table.items()}
    ambiguous = {k for k, v in table.items() if len({t for _, t in v}) > 1}
    info = {"rows": sum(len(v) for v in table.values()), "keys": len(table),
            "shared_keys": sum(len(v) > 1 for v in table.values()), "ambiguous_keys": len(ambiguous)}
    return out, {"ambiguous": ambiguous, **info}


class State:
    def __init__(self, mode: str, delay_s: float, slow_s: float, hang_s: float, seed: int, replay: dict,
                 replay_info: dict, retry_after_s: int = 1):
        self.mode, self.delay_s, self.slow_s, self.hang_s = mode, delay_s, slow_s, hang_s
        self.retry_after_s = retry_after_s
        self.rng = random.Random(seed)
        self.replay, self.replay_info = replay, replay_info
        self.lock = threading.Lock()
        self.calls: Counter = Counter()
        self.in_flight = 0
        self.max_in_flight = 0
        self.mode_changes: list[dict] = []
        self.t0 = time.monotonic()

    def snapshot(self) -> dict:
        with self.lock:
            info = {k: v for k, v in self.replay_info.items() if k != "ambiguous"}
            return {"mode": self.mode, "delay_s": self.delay_s, "calls": dict(self.calls),
                    "calls_total": sum(v for k, v in self.calls.items() if k.startswith("outcome:")),
                    "in_flight": self.in_flight, "max_in_flight": self.max_in_flight,
                    "replay": info, "mode_changes": self.mode_changes,
                    "uptime_s": round(time.monotonic() - self.t0, 3)}


def completion(content: str, model: str, prompt_chars: int) -> dict:
    """An OpenAI-style chat completion. Token counts are rough (characters / 4) so the service's cost code runs."""
    return {"id": f"fake-{time.time_ns()}", "object": "chat.completion", "created": int(time.time()),
            "model": model, "provider": "fake-upstream",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": max(1, prompt_chars // 4), "completion_tokens": max(1, len(content) // 4),
                      "total_tokens": max(1, prompt_chars // 4) + max(1, len(content) // 4),
                      "prompt_tokens_details": {"cached_tokens": 0}}}


def enums(body: dict) -> tuple[list[str], list[str]]:
    try:
        props = body["response_format"]["json_schema"]["schema"]["properties"]
        return list(props["intent"]["enum"]), list(props["action"]["enum"])
    except (KeyError, TypeError):
        return [], []


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "fake-upstream/1"

        def log_message(self, fmt, *args):          # quiet: the driver reads /_state instead
            pass

        def _send(self, code: int, payload: bytes, ctype: str = "application/json",
                  headers: dict | None = None) -> None:
            try:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(payload)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):      # the caller timed out and went away
                with state.lock:
                    state.calls["client_gone"] += 1

        def _json(self, code: int, obj, headers: dict | None = None) -> None:
            self._send(code, json.dumps(obj).encode(), headers=headers)

        def do_GET(self):
            if urlsplit(self.path).path == "/_state":
                return self._json(200, state.snapshot())
            return self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            u = urlsplit(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            if u.path == "/_mode":
                mode = (parse_qs(u.query).get("mode") or [""])[0]
                if mode not in MODES:
                    return self._json(400, {"error": f"mode must be one of {MODES}"})
                with state.lock:
                    state.mode_changes.append({"t_s": round(time.monotonic() - state.t0, 3), "from": state.mode,
                                               "to": mode, "unix": time.time()})
                    state.mode = mode
                return self._json(200, {"mode": mode})
            if not u.path.endswith("/chat/completions"):
                return self._json(404, {"error": {"message": "not found"}})
            with state.lock:
                state.in_flight += 1
                state.max_in_flight = max(state.max_in_flight, state.in_flight)
                mode = state.mode
                flip = state.rng.random() < 0.5
            try:
                self._chat(raw, mode, flip)
            finally:
                with state.lock:
                    state.in_flight -= 1

        def _count(self, outcome: str) -> None:
            with state.lock:
                state.calls[f"outcome:{outcome}"] += 1

        def _chat(self, raw: bytes, mode: str, flip: bool) -> None:
            try:
                body = json.loads(raw)
                msgs = body["messages"]
                user = next(m["content"] for m in msgs if m.get("role") == "user")
                model = str(body.get("model") or "fake")
            except Exception:                                    # noqa: BLE001
                self._count("bad_request")
                return self._json(400, {"error": {"message": "bad request body"}})
            if mode == "strict":
                bad = unknown_fields(body)
                if bad:
                    self._count("strict_rejected")
                    noun = "argument" if len(bad) == 1 else "arguments"
                    return self._json(400, {"error": {"message": f"Unrecognized request {noun} supplied: "
                                                                 + ", ".join(bad),
                                                      "type": "invalid_request_error", "param": bad[0],
                                                      "code": "unrecognized_argument"}})
            if mode == "ratelimit":
                self._count("ratelimit")
                wait = state.retry_after_s
                return self._json(429, {"error": {"message": f"Rate limit reached. Please try again in {wait}s.",
                                                  "type": "requests", "param": None, "code": "rate_limit_exceeded"}},
                                  headers={"Retry-After": str(wait)})
            if mode == "flaky":
                mode = "error500" if flip else "ok"
            if mode == "error500":
                self._count("error500")
                return self._json(500, {"error": {"message": "fake upstream: internal error", "code": 500}})
            if mode == "garbage":
                self._count("garbage")
                return self._send(200, b"Sorry, the model is overloaded right now. Please try again later.",
                                  "text/plain; charset=utf-8")
            intents, actions = enums(body)
            if mode == "garbage_content":
                self._count("garbage_content")
                text = "I think the customer wants help with their order, so the agent should look it up."
                return self._json(200, completion(text, model, len(user)))
            if mode == "bad_enum":
                self._count("bad_enum")
                text = json.dumps({"intent": intents[0] if intents else "x", "action": "escalate-to-manager",
                                   "needs_human": False, "confidence": 0.97})
                return self._json(200, completion(text, model, len(user)))
            if mode == "huge":
                self._count("huge")
                return self._json(200, completion("x" * 1_000_000, model, len(user)))
            wait = {"hang": state.hang_s, "slow": state.slow_s}.get(mode, state.delay_s)
            time.sleep(wait)
            if mode == "constant":
                self._count("constant")
                text = json.dumps({"intent": intents[0] if intents else "x", "action": "offer-refund",
                                   "needs_human": False, "confidence": 0.99})
                return self._json(200, completion(text, model, len(user)))
            key = key_from_prompt(user)
            text = state.replay.get(key) if key else None
            if text is not None:
                with state.lock:
                    state.calls["replay_hit"] += 1
                    if key in state.replay_info["ambiguous"]:
                        state.calls["replay_hit_ambiguous"] += 1
            else:
                with state.lock:
                    state.calls["replay_miss"] += 1
                act = next((a for a in FIXED_ACTIONS if a in actions), actions[0] if actions else "x")
                text = json.dumps({"intent": intents[0] if intents else "x", "action": act,
                                   "needs_human": False, "confidence": 0.9})
            self._count(mode)
            return self._json(200, completion(text, model, len(user)))

    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--mode", choices=MODES, default="ok")
    ap.add_argument("--delay-s", type=float, default=1.5, help="answer time in ok, flaky and constant modes")
    ap.add_argument("--slow-s", type=float, default=3.5)
    ap.add_argument("--hang-s", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0, help="seed for the flaky mode's coin")
    ap.add_argument("--retry-after-s", type=int, default=1, help="Retry-After seconds in ratelimit mode")
    ap.add_argument("--replay", help="recorded answers (JSONL with id and text), e.g. "
                                     "results/raw/llm-tail-luna_V3_none/raw_calls.jsonl")
    a = ap.parse_args(argv)
    replay, info = build_replay(Path(a.replay)) if a.replay else ({}, {"ambiguous": set()})
    state = State(a.mode, a.delay_s, a.slow_s, a.hang_s, a.seed, replay, info, a.retry_after_s)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(state))
    srv.daemon_threads = True
    print(json.dumps({"event": "fake_upstream_ready", "port": a.port, "mode": a.mode,
                      "replay": {k: v for k, v in info.items() if k != "ambiguous"}}), flush=True)
    try:
        srv.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
