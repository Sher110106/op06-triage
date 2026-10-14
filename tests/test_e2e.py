"""End to end over real HTTP: the service started as `scripts/reproduce.sh serve` starts it, with the LLM step off.

Each server is a subprocess on a free local port, with no API keys and no TRIAGE_* settings from the shell. One
server for the whole module answers the hostile pack, the parity rows, the malformed requests and the burst;
the policy reload and SIGTERM tests start their own.
"""
from __future__ import annotations

import concurrent.futures as cf
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from functools import cache
from pathlib import Path

import httpx
import pytest

from triage import text as T
from triage.classical import DEFAULT_BUNDLE
from triage.data import DATA, ROOT, project
from triage.policy import DEFAULT_POLICY, load

pytestmark = pytest.mark.skipif(not DEFAULT_BUNDLE.exists(),
                                reason="models/classical_v0.joblib is not built; run scripts/reproduce.sh train")

SCRIPT = ROOT / "scripts" / "reproduce.sh"
START_S = 120.0
ABSTAIN = {"hostile-007", "hostile-008"}
CLEAN_EXIT = (0, -signal.SIGTERM)       # uvicorn shuts down, then raises the signal again: the default exit
MASK_RULE = """
[[rules]]
name = "refund-needs-verification"
action = "offer-refund"
requires_any_observed = ["verify-identity", "validate-purchase"]
effect = "mask"
"""


@cache
def grader():
    spec = importlib.util.spec_from_file_location("op06_grader", DATA / "grader.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def encode(row) -> bytes:
    return row if isinstance(row, bytes) else json.dumps(row).encode("ascii")


class Server:
    """`scripts/reproduce.sh serve` in a subprocess. The script execs uvicorn, so the pid is the server's."""

    def __init__(self, log_dir: Path, port: int | None = None, policy: Path | None = None):
        self.port = port or free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.log = log_dir / f"server-{self.port}-{time.monotonic_ns()}.log"
        env = {k: v for k, v in os.environ.items() if not k.startswith("TRIAGE_") and "API_KEY" not in k}
        env.update(TRIAGE_LLM="off", HOST="127.0.0.1", PORT=str(self.port))
        if policy is not None:
            env["TRIAGE_POLICY_PATH"] = str(policy)
        with open(self.log, "wb") as out:
            self.proc = subprocess.Popen(["bash", str(SCRIPT), "serve"], cwd=ROOT, env=env, stdout=out,
                                         stderr=subprocess.STDOUT)
        self.client = httpx.Client(base_url=self.url, timeout=10, trust_env=False)
        deadline = time.monotonic() + START_S
        while True:
            if self.proc.poll() is not None:
                raise AssertionError(f"server exited with {self.proc.returncode}:\n{self.text()[-3000:]}")
            try:
                if self.client.get("/healthz").status_code == 200:
                    return
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                self.stop()
                raise AssertionError(f"no 200 from /healthz after {START_S} s:\n{self.text()[-3000:]}")
            time.sleep(0.2)

    def post(self, row, **kw) -> httpx.Response:
        headers = {"content-type": "application/json", **kw.pop("headers", {})}
        return self.client.post("/triage", content=encode(row), headers=headers, **kw)

    def health(self) -> dict:
        return self.client.get("/healthz").json()

    def text(self) -> str:
        return self.log.read_text(encoding="utf-8", errors="replace")

    def stop(self, timeout: float = 15.0) -> int:
        self.client.close()
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(5)
        return self.proc.returncode


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    s = Server(tmp_path_factory.mktemp("e2e"))
    yield s
    s.stop()


def test_ready_with_llm_off(server):
    h = server.health()
    assert h["status"] == "ok" and h["llm"]["enabled"] is False
    assert h["policy"]["version"] == load(DEFAULT_POLICY).version and h["policy"]["last_error"] is None
    assert "LLM step: off" in server.text()


def test_hostile_pack(server, triage_off, split):
    for row in map(project, split("hostile")):
        t = time.perf_counter()
        resp = server.post(row, timeout=2.0)
        assert resp.status_code == 200 and time.perf_counter() - t < 2.0, row["id"]
        body = resp.json()
        assert grader().schema_valid(body, None) and body["id"] == row["id"]
        assert body["needs_human"] is True, row["id"]
        if row["id"] in ABSTAIN:
            assert (body["intent"], body["action"], body["confidence"]) == ("unknown", "none", 0.0)
            assert body["error"]["code"] == "invalid_input"
        else:
            assert "error" not in body and body["confidence"] <= 0.5, row["id"]
        assert body == triage_off.handle(row, "e2e")[0]


def test_dev_rows_match_in_process(server, triage_off, split):
    rows = [project(r) for r in split("dev")[::40]]
    assert len(rows) == 50
    for row in rows:
        rid = "e2e-" + row["id"]
        resp = server.post(row, headers={"X-Request-ID": rid})
        assert resp.status_code == 200 and resp.headers["x-request-id"] == rid
        assert resp.json() == triage_off.handle(row, rid)[0], row["id"]


PROBES = [  # name, body, HTTP status, error code
    ("bad JSON", b'{"id": "x", "context": [', 400, "bad_json"),
    ("not UTF-8", b'{"id": "\xff", "context": []}', 400, "bad_json"),
    ("NaN", b'{"id": "x", "context": [], "n": NaN}', 400, "bad_json"),
    ("too deep", b'{"id": "x", "context": [], "n": ' + b"[" * 100 + b"]" * 100 + b"}", 400, "bad_json"),
    ("array", b"[1, 2]", 400, "invalid_envelope"),
    ("missing id", b'{"context": [{"speaker": "customer", "text": "hi"}]}', 400, "invalid_envelope"),
    ("empty id", b'{"id": "", "context": []}', 400, "invalid_envelope"),
    ("id not text", b'{"id": 7, "context": []}', 400, "invalid_envelope"),
    ("2 MiB body", b'{"id": "big", "context": [{"speaker": "customer", "text": "' + b"a" * (2 << 20) + b'"}]}',
     413, "body_too_large"),
]


@pytest.mark.parametrize("name,body,status,code", PROBES, ids=[p[0] for p in PROBES])
def test_malformed_requests(server, name, body, status, code):
    resp = server.post(body)
    assert resp.status_code == status
    err = resp.json()["error"]
    assert err["code"] == code and isinstance(err["message"], str) and err["message"]
    assert resp.headers["x-request-id"]
    assert server.client.get("/healthz").status_code == 200


@pytest.mark.parametrize("raw_id", [b"x" * 5000, b"\\ud800", b"caf\\u00e9"])
def test_any_non_empty_id_is_answered(server, raw_id):
    """input.schema.json asks only for a non-empty string id: long or odd ids get a 200 that echoes them."""
    resp = server.post(b'{"id": "' + raw_id + b'", "context": []}')
    body = resp.json()
    assert resp.status_code == 200 and grader().schema_valid(body, None)
    assert body["id"] == json.loads(b'"' + raw_id + b'"') and body["error"]["code"] == "invalid_input"


def test_chunked_body_over_limit(server):
    def chunks():
        yield b'{"id": "big", "context": [{"speaker": "customer", "text": "'
        for _ in range(17):
            yield b"a" * 65536
        yield b'"}]}'

    resp = server.client.post("/triage", content=chunks(), headers={"content-type": "application/json"})
    assert resp.status_code == 413 and resp.json()["error"]["code"] == "body_too_large"


def test_100_requests_at_once(server, triage_off, split):
    """100 clients, one connection each, released together."""
    rows = [project(r) for r in split("dev")[1000:1100]]
    expected = {r["id"]: triage_off.handle(r, "e2e")[0] for r in rows}
    start = threading.Barrier(len(rows))

    def one(row):
        with httpx.Client(base_url=server.url, timeout=30, trust_env=False) as c:
            start.wait(30)
            t = time.perf_counter()
            r = c.post("/triage", content=encode(row),
                       headers={"content-type": "application/json", "X-Request-ID": "burst-" + row["id"]})
            return row["id"], r, time.perf_counter() - t

    with cf.ThreadPoolExecutor(len(rows)) as ex:
        out = list(ex.map(one, rows))
    assert [r.status_code for _, r, _ in out] == [200] * len(rows)
    for rid, r, _ in out:
        assert r.json() == expected[rid] and r.headers["x-request-id"] == "burst-" + rid
    assert max(s for *_, s in out) < 10


def test_logs_one_json_line_per_request_without_text(server, split):
    rows = [project(r) for r in split("hostile") + split("dev")[:5]]
    for row in rows:
        assert server.post(dict(row, id="log-" + row["id"])).status_code == 200
    requests = server.client.get("/metrics").json()["requests"]
    text = server.text()
    lines = [json.loads(x) for x in text.splitlines() if x.startswith("{") and '"request_id"' in x]
    assert len(lines) == requests
    for row in rows:
        for t in row["context"]:
            if t["speaker"] != "action" and len(t["text"]) >= 20:
                assert t["text"] not in text


def test_policy_hot_reload(tmp_path, triage_off, split):
    rows = [project(r) for r in split("dev")]
    need = {"verify-identity", "validate-purchase"}
    target = next(r for r, (resp, _) in zip(rows, triage_off.handle_many(rows))
                  if resp["action"] == "offer-refund" and not need & set(T.observed_actions(r)))
    policy = tmp_path / "policy.toml"
    shutil.copy(DEFAULT_POLICY, policy)
    s = Server(tmp_path, policy=policy)
    try:
        old = s.health()["policy"]["version"]
        assert s.post(target).json()["action"] == "offer-refund"
        text = policy.read_text()
        assert f'version = "{old}"' in text
        policy.write_text(text.replace(f'version = "{old}"', 'version = "e2e-mask.1"') + MASK_RULE)
        time.sleep(2.1)                                       # the service looks at the file at most every 2 s
        deadline = time.monotonic() + 10
        while (h := s.health())["policy"]["version"] != "e2e-mask.1":
            assert time.monotonic() < deadline, h["policy"]
            time.sleep(0.2)
        assert h["policy"]["last_error"] is None
        after = s.post(target)                                # the same body: the reload empties the cache
        assert after.status_code == 200 and grader().schema_valid(after.json(), None)
        assert after.json()["action"] != "offer-refund" and "x-idempotent-replay" not in after.headers
        assert s.client.get("/metrics").json()["policy_version"] == "e2e-mask.1"

        policy.write_text('version = "e2e-broken"\n[flows\n')            # a broken save keeps the last good policy
        time.sleep(2.1)
        deadline = time.monotonic() + 10
        while (h := s.health())["policy"]["last_error"] is None:
            assert time.monotonic() < deadline, h["policy"]
            time.sleep(0.2)
        assert h["policy"]["version"] == "e2e-mask.1" and "TOML" in h["policy"]["last_error"]
        still = s.post(dict(target, id=target["id"] + "-broken"))
        assert still.status_code == 200 and still.json()["action"] == after.json()["action"]
    finally:
        s.stop()


def test_sigterm_stops_cleanly_and_restart_serves(tmp_path, split):
    row = project(split("dev")[0])
    s = Server(tmp_path)
    assert s.post(row).status_code == 200
    t = time.monotonic()
    assert s.stop() in CLEAN_EXIT and time.monotonic() - t < 10
    assert "Application shutdown complete" in s.text() and "Finished server process" in s.text()
    s2 = Server(tmp_path, port=s.port)                        # the same port is free again
    try:
        r = s2.post(row)
        assert r.status_code == 200 and r.json()["id"] == row["id"]
    finally:
        assert s2.stop() in CLEAN_EXIT
