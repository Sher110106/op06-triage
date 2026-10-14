"""Fault injection: the real service (uvicorn, the DESIGN.md serve flags) against a fake LLM upstream that fails in
set ways, plus 10x load, a kill -9 restart and SIGTERM drains. No API key and no network beyond 127.0.0.1.

  .venv/bin/python experiments/chaos/run_chaos.py all            every step below, in order (about 20 min on an M1)
  .venv/bin/python experiments/chaos/run_chaos.py ref-replay     in-process reference: recorded Luna answers replayed
  .venv/bin/python experiments/chaos/run_chaos.py load NAME      one load scenario (see SCENARIOS)
  .venv/bin/python experiments/chaos/run_chaos.py recovery       upstream 500s for 20 s, then healthy: does it recover?
  .venv/bin/python experiments/chaos/run_chaos.py restart        kill -9 under traffic, restart, SIGTERM drains
  .venv/bin/python experiments/chaos/run_chaos.py report         chaos.json and the markdown tables

Inputs for every load scenario: the first 300 dev rows and all 300 dev_noisy rows (600), projected with
triage.data.project, sent by scripts/loadtest.py with --warmup 0 (the breaker starts closed in the measured run).
Each scenario writes results/raw/chaos-<name>/: predictions.jsonl, latency_bench.json, service.log (the service's
JSON lines and uvicorn's own lines; no customer text), fake_upstream.log, health_timeline.json, meta.json,
summary.json, and the harness report (report.json, report_official.json; graded with the 1,700 unsent rows
excluded, so read report.json "extras"; report_official.json counts unsent rows as wrong).

The service's API key is "test-key" and its base URL is the fake upstream on 127.0.0.1; every variable whose name
looks like a key, token or secret is removed from the child processes' environment.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from harness.eval import sha256  # noqa: E402
from triage.data import DATA, SPLITS, load_split, project  # noqa: E402

os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
PY = str(ROOT / ".venv/bin/python")
SERVE_ARGS = ["--workers", "1", "--limit-concurrency", "256", "--timeout-keep-alive", "5", "--no-access-log"]
REPLAY = ROOT / "results/raw/llm-tail-luna_V3_none/raw_calls.jsonl"
RAW = ROOT / "results/raw"
WORK = Path(tempfile.gettempdir()) / "op06-chaos"
LOOP = "127.0.0.1"
DEGRADED = ("timeout", "http_error", "invalid", "breaker_open", "busy")
FIELDS = ("id", "intent", "action", "confidence", "needs_human")
N_DEV, N_NOISY = 300, 300

# name -> (fake upstream mode or None, concurrency, LLM on); "down" points the service at a closed port
SCENARIOS = {
    "cpu-ref": (None, 8, False),
    "ok": ("ok", 8, True),
    "error500": ("error500", 8, True),
    "hang": ("hang", 8, True),
    "garbage": ("garbage", 8, True),
    "garbage_content": ("garbage_content", 8, True),
    "bad_enum": ("bad_enum", 8, True),
    "slow": ("slow", 8, True),
    "flaky": ("flaky", 8, True),
    "constant": ("constant", 8, True),
    "down": ("down", 8, True),
    "huge": ("huge", 8, True),
    "cpu-c80": (None, 80, False),
    "ok-c80": ("ok", 80, True),
    "ok-c80-r2": ("ok", 80, True),           # repeats: the laptop is shared, so one run is not enough
    "ok-c80-r3": ("ok", 80, True),
    "slow-c80": ("slow", 80, True),
    "cpu-c128": (None, 128, False),          # just under the 128-request shed point
    "ok-c200": ("ok", 200, True),
    "cpu-c200": (None, 200, False),
    "ok-c200x4": ("ok", 200, True),          # the same 200 clients from 4 load-test processes of 50
}
PROCS = {"ok-c200x4": 4}
ORDER = list(SCENARIOS)


# ----------------------------------------------------------------------------- helpers

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def free_port() -> int:
    with socket.socket() as s:
        s.bind((LOOP, 0))
        return s.getsockname()[1]


def child_env(extra: dict | None = None) -> dict:
    """The parent environment without anything that looks like a credential, a proxy or a TRIAGE_ setting."""
    bad = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|PROXY", re.I)
    env = {k: v for k, v in os.environ.items() if not bad.search(k) and not k.startswith("TRIAGE_")}
    env.update(NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
    env.update(extra or {})
    return env


def llm_env(base_url: str | None) -> dict:
    if base_url is None:
        return child_env({"TRIAGE_LLM": "off"})
    return child_env({"TRIAGE_LLM": "on", "TRIAGE_LLM_BASE_URL": base_url, "TRIAGE_LLM_API_KEY": "test-key"})


def get_json(url: str, timeout: float = 5.0) -> dict:
    try:
        r = httpx.get(url, timeout=timeout)
        try:
            body = r.json()
        except ValueError:
            body = {"text": r.text[:200]}
        return {"http_status": r.status_code, **(body if isinstance(body, dict) else {"body": body})}
    except Exception as e:                                        # noqa: BLE001
        return {"error": type(e).__name__}


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def read_preds(path: Path) -> dict:
    if not path.exists():
        return {}
    return {p["id"]: p for p in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if p}


def same(a: dict | None, b: dict | None) -> bool:
    return a is not None and b is not None and all(a.get(k) == b.get(k) for k in FIELDS)


def all_rows() -> list[dict]:
    return [project(r) for r in load_split("dev")] + [project(r) for r in load_split("noisy")]


def inputs_file() -> tuple[Path, list[str]]:
    rows = [project(r) for r in load_split("dev")[:N_DEV]] + [project(r) for r in load_split("noisy")[:N_NOISY]]
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / "inputs_600.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path, [r["id"] for r in rows]


def exclude_file(ids: set[str]) -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    rest = sorted(r["id"] for s in ("dev", "noisy") for r in load_split(s) if r["id"] not in ids)
    path = WORK / f"exclude_{len(rest)}.json"
    path.write_text(json.dumps(rest))
    return path


def gold() -> dict:
    return {r["id"]: r for s in ("dev", "noisy") for r in load_split(s)}


def grade(out: Path, ids: set[str]) -> dict | None:
    pred = out / "predictions.jsonl"
    if not pred.exists() or not pred.read_text().strip():
        return None
    r = subprocess.run([PY, "-m", "harness.eval", "grade", "--pred", str(pred), "--out", str(out / "report.json"),
                        "--exclude-ids", str(exclude_file(ids))], cwd=ROOT, env=child_env(), capture_output=True,
                       text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:], file=sys.stderr)
        return None
    rep = json.loads((out / "report.json").read_text())
    return {"rows_scored": rep["extras"]["rows"], **rep["extras"]["point"],
            "needs_human_rate": rep["extras"]["needs_human_rate"],
            "action_acc_when_not_routed": rep["extras"]["action_acc_when_not_routed"],
            "ci95_action_acc": rep["extras"]["ci95_conversation_bootstrap"]["action_acc"]}


def inputs_meta(ids: list[str], path: Path) -> dict:
    return {"rows": len(ids), "recipe": f"first {N_DEV} rows of dev, then the first {N_NOISY} rows of dev_noisy, "
                                        "projected with triage.data.project",
            "inputs_sha256": sha256(path), "first_id": ids[0], "last_id": ids[-1],
            "data": {k: {"file": f"data/op06/{SPLITS[k]}", "sha256": sha256(DATA / SPLITS[k])}
                     for k in ("dev", "noisy")}}


def machine() -> dict:
    return {"python": platform.python_version(), "machine": f"{platform.system()} {platform.machine()}",
            "cpu_count": os.cpu_count(), "load_avg": [round(x, 2) for x in os.getloadavg()]}


# ----------------------------------------------------------------------------- processes

class Fake:
    """experiments/chaos/fake_upstream.py in a child process, with the recorded Luna answers for replay."""

    def __init__(self, mode: str, log_path: Path, **opts):
        self.port = free_port()
        self.url = f"http://{LOOP}:{self.port}"
        cmd = [PY, str(HERE / "fake_upstream.py"), "--port", str(self.port), "--mode", mode, "--replay", str(REPLAY)]
        for k, v in opts.items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        self.log = open(log_path, "w", encoding="utf-8")
        self.p = subprocess.Popen(cmd, cwd=ROOT, env=child_env(), stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 120
        while get_json(self.url + "/_state", 1.0).get("http_status") != 200:
            if self.p.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"fake upstream did not start; see {log_path}")
            time.sleep(0.2)

    def state(self) -> dict:
        return get_json(self.url + "/_state")

    def set_mode(self, mode: str) -> None:
        httpx.post(f"{self.url}/_mode?mode={mode}", timeout=5).raise_for_status()

    def stop(self) -> None:
        self.p.terminate()
        try:
            self.p.wait(10)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()
        self.log.close()


class Svc:
    """The service as reproduce.sh serve runs it (host 127.0.0.1 instead of 0.0.0.0)."""

    def __init__(self, env: dict, log_path: Path, port: int | None = None):
        self.env, self.log_path = env, log_path
        self.port = port or free_port()
        self.url = f"http://{LOOP}:{self.port}"
        self.p: subprocess.Popen | None = None

    def start(self) -> float:
        self.log = open(self.log_path, "a", encoding="utf-8")
        self.t_spawn = time.monotonic()
        self.p = subprocess.Popen([PY, "-m", "uvicorn", "triage.service:app", "--host", LOOP, "--port", str(self.port),
                                   *SERVE_ARGS], cwd=ROOT, env=self.env, stdout=self.log, stderr=subprocess.STDOUT)
        return self.t_spawn

    def wait_healthy(self, timeout: float = 180.0, poll: float = 0.1) -> dict:
        """Seconds from spawn to the first HTTP answer on /healthz and to the first 200."""
        first_answer, seen = None, Counter()
        while True:
            h = get_json(self.url + "/healthz", 2.0)
            t = time.monotonic() - self.t_spawn
            seen[str(h.get("http_status", h.get("error")))] += 1
            if "http_status" in h and first_answer is None:
                first_answer = t
            if h.get("http_status") == 200:
                return {"listening_s": round(first_answer, 3), "healthy_s": round(t, 3), "polls": dict(seen),
                        "health": h}
            if self.p.poll() is not None:
                raise RuntimeError(f"service exited with {self.p.returncode} during start-up; see {self.log_path}")
            if t > timeout:
                raise RuntimeError(f"no 200 from /healthz after {timeout} s")
            time.sleep(poll)

    def stop(self, sig=signal.SIGTERM, timeout: float = 60.0, on_signal=None) -> dict:
        t0 = time.monotonic()
        self.p.send_signal(sig)
        if on_signal is not None:
            on_signal()
        try:
            rc = self.p.wait(timeout)
        except subprocess.TimeoutExpired:
            self.p.kill()
            rc = self.p.wait()
            return {"signal": signal.Signals(sig).name, "exit_code": rc, "exit_s": None, "forced_kill": True}
        finally:
            self.log.flush()
        return {"signal": signal.Signals(sig).name, "exit_code": rc, "exit_s": round(time.monotonic() - t0, 3),
                "forced_kill": False}


class Poller(threading.Thread):
    """GET /healthz every `every` s; keeps (t, http status, breaker)."""

    def __init__(self, url: str, every: float = 0.5, t0: float | None = None):
        super().__init__(daemon=True)
        self.url, self.every = url, every
        self.t0 = t0 if t0 is not None else time.monotonic()
        self.samples: list[dict] = []
        self.halt = threading.Event()

    def run(self):
        while not self.halt.is_set():
            t = time.monotonic()
            h = get_json(self.url + "/healthz", 10.0)
            t_h = time.monotonic()
            m = get_json(self.url + "/metrics", 10.0)              # answered on the event loop, no thread hop
            t_m = time.monotonic()
            llm = h.get("llm") if isinstance(h.get("llm"), dict) else {}
            self.samples.append({"t_s": round(t - self.t0, 3), "http": h.get("http_status", h.get("error")),
                                 "breaker": llm.get("breaker"), "healthz_ms": round((t_h - t) * 1000, 1),
                                 "metrics_ms": round((t_m - t_h) * 1000, 1) if "http_status" in m else None})
            self.halt.wait(max(0.0, self.every - (time.monotonic() - t)))

    def stop(self) -> list[dict]:
        self.halt.set()
        self.join(5)
        return self.samples

    @staticmethod
    def rtt(samples: list[dict]) -> dict:
        """Round trips of the probe, a separate client: /metrics runs on the service's event loop, so a slow
        /metrics means the loop (or the process) is stalled, not the load client."""
        out = {}
        for k in ("healthz_ms", "metrics_ms"):
            v = sorted(x[k] for x in samples if x.get(k) is not None)
            if v:
                out[k] = {"n": len(v), "p50": v[len(v) // 2], "max": v[-1]}
        return out

    @staticmethod
    def transitions(samples: list[dict]) -> list[dict]:
        out, last = [], object()
        for s in samples:
            key = (s["http"], s["breaker"])
            if key != last:
                out.append(s)
                last = key
        return out


class Driver:
    """POST /triage from `workers` threads, each at most `rate` requests/s, until stopped. Keeps every outcome."""

    def __init__(self, url: str, rows: list[dict], workers: int = 4, rate: float = 4.0, timeout: float = 30.0,
                 t0: float | None = None):
        self.url, self.rows, self.workers, self.rate, self.timeout = url, rows, workers, rate, timeout
        self.t0 = t0 if t0 is not None else time.monotonic()
        self.recs: list[dict] = []
        self.lock = threading.Lock()
        self.halt = threading.Event()
        self.next = 0
        self.in_flight: dict[str, float] = {}
        self.threads = [threading.Thread(target=self._work, daemon=True) for _ in range(workers)]

    def start(self) -> "Driver":
        for t in self.threads:
            t.start()
        return self

    def _row(self):
        with self.lock:
            k = self.next
            self.next += 1
        row = self.rows[k % len(self.rows)]
        if k >= len(self.rows):                                    # a second pass: new id, so no cache hit
            row = {**row, "id": f"{row['id']}-pass{k // len(self.rows)}"}
        return k, row

    def _work(self):
        client = httpx.Client(base_url=self.url, timeout=self.timeout)
        due = time.monotonic()
        while not self.halt.is_set():
            seq, row = self._row()
            rid = f"chaos-{uuid.uuid4().hex[:16]}"
            start = time.monotonic()
            with self.lock:
                self.in_flight[rid] = start
            status, body, err = None, None, None
            try:
                r = client.post("/triage", content=json.dumps(row).encode("ascii"),
                                headers={"Content-Type": "application/json", "X-Request-ID": rid})
                status = r.status_code
                try:
                    body = r.json()
                except ValueError:
                    body = {"text": r.text[:100]}
            except Exception as e:                                 # noqa: BLE001
                err = type(e).__name__
            end = time.monotonic()
            with self.lock:
                self.in_flight.pop(rid, None)
                self.recs.append({"seq": seq, "id": row["id"], "request_id": rid,
                                  "t_start": round(start - self.t0, 4), "t_end": round(end - self.t0, 4),
                                  "status": status, "error": err,
                                  "error_code": (body.get("error") or {}).get("code")
                                  if isinstance(body, dict) and status != 200 and isinstance(body.get("error"), dict)
                                  else None,
                                  "body": body if status == 200 else None})
            due = max(due + 1.0 / self.rate, time.monotonic() - 0.05)
            self.halt.wait(max(0.0, due - time.monotonic()))
        client.close()

    def outstanding(self) -> list[float]:
        with self.lock:
            return sorted(self.in_flight.values())

    def stop(self) -> list[dict]:
        self.halt.set()
        for t in self.threads:
            t.join(self.timeout + 5)
        return sorted(self.recs, key=lambda r: r["t_start"])


def read_log(path: Path) -> tuple[dict, list[str]]:
    """request_id -> the service's JSON log line; and every line that is not a request line."""
    by_rid, other = {}, []
    if not path.exists():
        return by_rid, other
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            other.append(line)
            continue
        if isinstance(rec, dict) and "request_id" in rec:
            by_rid[rec["request_id"]] = rec
        else:
            other.append(line)
    return by_rid, other


def clean_stop(stop: dict, log: dict) -> bool:
    """uvicorn 0.54 finishes its shutdown, then raises the caught signal again, so a clean SIGTERM stop ends with
    status -15 (143 in a shell) after "Application shutdown complete" and "Finished server process"."""
    return (not stop.get("forced_kill") and stop.get("exit_code") in (0, -signal.SIGTERM)
            and log["shutdown_complete"] and log["finished_server_process"] and log["tracebacks"] == 0)


def log_health(other: list[str]) -> dict:
    text = "\n".join(other)
    return {"tracebacks": text.count("Traceback"), "error_lines": sum("ERROR" in x for x in other),
            "shutdown_complete": "Application shutdown complete" in text,
            "finished_server_process": "Finished server process" in text,
            "load_failed": "load_failed" in text}


# ----------------------------------------------------------------------------- reference runs

def cmd_ref_replay(_a=None) -> dict:
    """In-process pipeline with the recorded Luna answers replayed (the Replay pattern of
    experiments/routing/analyse_routing.py), per request, over the 600 inputs: what the "ok" mode should return."""
    from triage import pipeline as P
    from triage.llm_action import LLMConfig, LLMResult

    class Replay:
        def __init__(self, path, gate):
            self.cfg = LLMConfig(mode="on", api_key="replay", gate=gate)
            self.rec = {}
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                p = r.get("pred") or {}
                ok = r.get("status") == "ok" and r.get("parse_status") in ("ok", "lenient_ok") and p
                status = "invalid" if not ok else (
                    "abstained" if p.get("intent") == "unknown" or p.get("action") == "none" else "ok")
                self.rec[r["id"]] = LLMResult(status, intent=p.get("intent"), action=p.get("action"))

        def decide(self, row):
            return self.rec.get(row["id"], LLMResult("http_error", error="no recorded answer"))

        def close(self):
            pass

    out = RAW / "chaos-ref-replay"
    out.mkdir(parents=True, exist_ok=True)
    path, ids = inputs_file()
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    tr = P.Triage(llm_cfg=LLMConfig(mode="off"))
    tr.llm = Replay(REPLAY, LLMConfig().gate)
    status = Counter()
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as fh, \
            open(out / "traces.jsonl", "w", encoding="utf-8") as th:
        for r in rows:
            resp, trace = tr.handle(r, uuid.uuid4().hex)
            status[(trace.get("llm") or {}).get("status")] += 1
            fh.write(json.dumps(resp, ensure_ascii=False) + "\n")
            th.write(json.dumps({"id": r["id"], **trace}, ensure_ascii=False, default=str) + "\n")
    h = tr.health()
    tr.close()
    rep = grade(out, set(ids))
    meta = {"run_id": out.name, "created": now_iso(), "system": "pipeline in-process, Triage.handle per row, LLM step "
            "replaced by the recorded GPT-6 Luna answers (no API call)", "llm_answers": str(REPLAY.relative_to(ROOT)),
            "changed": "LLM step replaced by a replay of recorded answers; nothing else",
            "bundle": h["bundle"], "bundle_sha256": h["bundle_sha256"], "policy": h["policy"],
            "inputs": inputs_meta(ids, path), "llm_status": {str(k): v for k, v in status.items()},
            "graded": rep, "grading_note": "graded with the 1,700 unsent rows excluded; read report.json extras",
            **machine()}
    write_json(out / "meta.json", meta)
    print(json.dumps({"run": out.name, "llm_status": meta["llm_status"], "graded": rep}))
    return meta


# ----------------------------------------------------------------------------- load scenarios

def analyse_load(out: Path, bench: dict, ids: list[str]) -> dict:
    cpu = read_preds(RAW / "chaos-cpu-ref" / "predictions.jsonl")
    rep = read_preds(RAW / "chaos-ref-replay" / "predictions.jsonl")
    preds = read_preds(out / "predictions.jsonl")
    by_rid, other = read_log(out / "service.log")
    g = gold()
    groups: dict[str, list[str]] = {}
    statuses = Counter()
    unlogged = 0
    for r in bench["requests"]:
        line = by_rid.get(r["request_id"])
        if line is None:
            unlogged += 1
        st = line.get("llm_status") if line else None
        if r["status"] != 200:
            st = f"http_{r['status']}" if r["status"] is not None else "client_error"
        statuses[str(st)] += 1
        groups.setdefault(str(st), []).append(r["id"])
    by_group = {}
    for st, gids in sorted(groups.items()):
        by_group[st] = {"rows": len(gids),
                        "equal_cpu_only": sum(same(preds.get(i), cpu.get(i)) for i in gids) if cpu else None,
                        "equal_replay_ref": sum(same(preds.get(i), rep.get(i)) for i in gids) if rep else None}
    failed = [i for st in DEGRADED for i in groups.get(st, [])]
    changed = [i for i in ids if i in preds and i in cpu and preds[i]["action"] != cpu[i]["action"]]
    answered = [i for i in ids if i in preds]
    acc = lambda key: round(sum(preds[i][key] == g[i][key] for i in answered) / len(answered), 4) if answered else None
    auto = [i for i in answered if not preds[i]["needs_human"]]
    return {
        "llm_status_by_row": dict(statuses), "rows_by_status": by_group,
        "llm_failed_rows": len(failed),
        "llm_failed_equal_cpu_only": sum(same(preds.get(i), cpu.get(i)) for i in failed) if cpu else None,
        "rows_equal_cpu_only": sum(same(preds.get(i), cpu.get(i)) for i in ids) if cpu else None,
        "rows_equal_replay_ref": sum(same(preds.get(i), rep.get(i)) for i in ids) if rep else None,
        "action_changed_vs_cpu": len(changed),
        "action_changed_and_routed": sum(bool(preds[i]["needs_human"]) for i in changed),
        "answered": len(answered), "intent_acc": acc("intent"), "action_acc": acc("action"),
        "routed_share": round(sum(bool(preds[i]["needs_human"]) for i in answered) / len(answered), 4)
        if answered else None,
        "auto_action_acc": round(sum(preds[i]["action"] == g[i]["action"] for i in auto) / len(auto), 4)
        if auto else None,
        "requests_not_in_log": unlogged, "log": log_health(other)}


def run_loadtest(url: str, conc: int, path: Path, out: Path, procs: int = 1):
    """scripts/loadtest.py once, or `procs` copies at once over interleaved slices of the inputs with conc/procs
    clients each (one Python client process tops out near 200 connections), merged into one latency_bench.json."""
    base = [PY, "scripts/loadtest.py", "--url", url, "--warmup", "0", "--wait-s", "10", "--label", out.name]
    if procs == 1:
        return subprocess.run(base + ["--concurrency", str(conc), "--inputs", str(path),
                                      "--out", str(out / "latency_bench.json"),
                                      "--pred-out", str(out / "predictions.jsonl")],
                              cwd=ROOT, env=child_env(), capture_output=True, text=True)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    parts = []
    for k in range(procs):
        sl = WORK / f"slice_{k}.jsonl"
        sl.write_text("".join(lines[k::procs]), encoding="utf-8")
        parts.append((sl, WORK / f"bench_{k}.json", WORK / f"pred_{k}.jsonl"))
    t0 = time.monotonic()
    ps = [subprocess.Popen(base + ["--concurrency", str(conc // procs), "--inputs", str(sl), "--out", str(bo),
                                   "--pred-out", str(po)], cwd=ROOT, env=child_env(), stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True) for sl, bo, po in parts]
    outs = [p.communicate()[0] for p in ps]
    wall = time.monotonic() - t0
    benches = [json.loads(bo.read_text()) for _, bo, _ in parts]
    reqs = []
    for k, b in enumerate(benches):
        reqs += [{**r, "proc": k} for r in b["requests"]]
    import numpy as np
    lat = np.asarray([r["duration_ms"] for r in reqs if r["status"] is not None])
    p50, p95, p99 = (round(float(x), 3) for x in np.percentile(lat, [50, 95, 99]))
    first, last = benches[0], benches
    merged = {"label": out.name, "created": first["created"], "url": url, "concurrency": conc, "processes": procs,
              "sample_count": len(reqs), "repeat": 1,
              **{k: sum(b[k] for b in benches) for k in ("errors", "non_200", "schema_invalid", "id_mismatch",
                                                         "replays")},
              "latency_ms": {"p50": p50, "p95": p95, "p99": p99, "max": round(float(lat.max()), 3),
                             "mean": round(float(lat.mean()), 3)},
              "throughput_rps": round(len(reqs) / wall, 2), "wall_s": round(wall, 3),
              "per_process": [{k: b[k] for k in ("sample_count", "errors", "non_200", "latency_ms", "throughput_rps",
                                                 "wall_s")} for b in last],
              "client": first["client"], "server_health": first["server_health"],
              "llm_calls": sum(b["llm_calls"] for b in benches), "requests": reqs}
    write_json(out / "latency_bench.json", merged)
    (out / "predictions.jsonl").write_text("".join(po.read_text(encoding="utf-8") for _, _, po in parts),
                                           encoding="utf-8")
    return subprocess.CompletedProcess(args="loadtest x%d" % procs, returncode=max(p.returncode for p in ps),
                                       stdout="\n".join(o.strip()[-300:] for o in outs), stderr="")


def cmd_load(a) -> dict:
    name = a.name
    mode, conc, llm = SCENARIOS[name]
    out = RAW / f"chaos-{name}"
    out.mkdir(parents=True, exist_ok=True)
    for f in ("service.log", "fake_upstream.log", "predictions.jsonl"):
        (out / f).unlink(missing_ok=True)
    path, ids = inputs_file()
    fake = Fake(mode, out / "fake_upstream.log") if llm and mode != "down" else None
    if not llm:
        base = None
    elif fake is not None:
        base = f"{fake.url}/v1"
    else:
        base = f"http://{LOOP}:{free_port()}/v1"                  # nothing listens there
    svc = Svc(llm_env(base), out / "service.log")
    svc.start()
    t_start = svc.t_spawn
    try:
        start = svc.wait_healthy()
        poller = Poller(svc.url, 0.25)
        poller.start()
        load_before = machine()["load_avg"]
        ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)        # load-test processes only: the service and
        lt = run_loadtest(svc.url, conc, path, out, PROCS.get(name, 1))   # the fake are still running
        ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        client_cpu_s = round(ru1.ru_utime + ru1.ru_stime - ru0.ru_utime - ru0.ru_stime, 3)
        samples = poller.stop()
        health, metrics = get_json(svc.url + "/healthz"), get_json(svc.url + "/metrics")
        fstate = fake.state() if fake else None
    finally:
        stop = svc.stop()
        if fake:
            fake.stop()
    bench = json.loads((out / "latency_bench.json").read_text())
    write_json(out / "health_timeline.json", {"every_s": 0.25, "samples": samples,
                                              "transitions": Poller.transitions(samples)})
    summary = {"run_id": out.name, "mode": mode, "concurrency": conc, "llm": llm,
               "loadtest_exit_code": lt.returncode, "loadtest_stdout": lt.stdout.strip()[-1500:],
               "sample_count": bench["sample_count"], "errors": bench["errors"], "non_200": bench["non_200"],
               "schema_invalid": bench["schema_invalid"], "id_mismatch": bench["id_mismatch"],
               "status_counts": dict(Counter(str(r["status"]) for r in bench["requests"])),
               "error_codes": dict(Counter(r["error"] for r in bench["requests"] if r["error"])),
               "latency_ms": bench["latency_ms"], "throughput_rps": bench["throughput_rps"], "wall_s": bench["wall_s"],
               "llm_calls": bench["llm_calls"], "client_cpu_s": client_cpu_s,
               "metrics_after": {k: metrics.get(k) for k in ("requests", "by_status", "errors", "routed", "llm",
                                                               "latency_ms")},
               "breaker_after": (health.get("llm") or {}).get("breaker"),
               "breaker_seen": sorted({str(s["breaker"]) for s in samples}),
               "breaker_first_not_closed_s": next((s["t_s"] for s in samples
                                                   if s["breaker"] not in (None, "closed")), None),
               "healthz_non_200_during_run": sum(s["http"] != 200 for s in samples),
               "probe_rtt_ms": Poller.rtt(samples),
               "fake_upstream": fstate, "start": {k: start[k] for k in ("listening_s", "healthy_s", "polls")},
               "stop": stop, **analyse_load(out, bench, ids)}
    summary["log"]["shutdown_clean"] = clean_stop(stop, summary["log"])
    summary["graded"] = grade(out, set(ids))
    write_json(out / "summary.json", summary)
    h = start["health"]
    changed = {"cpu-ref": "TRIAGE_LLM=off (reference: the CPU-only answer and latency)",
               "down": "TRIAGE_LLM_BASE_URL points at a 127.0.0.1 port with nothing listening"}.get(
        name, f"TRIAGE_LLM_BASE_URL points at experiments/chaos/fake_upstream.py in mode {mode!r}")
    meta = {"run_id": out.name, "created": now_iso(), "system": "service-http (uvicorn, DESIGN.md serve flags, "
            "host 127.0.0.1)", "changed": changed, "concurrency": conc,
            "service_env": {"TRIAGE_LLM": "on" if llm else "off", "TRIAGE_LLM_BASE_URL": base,
                            "TRIAGE_LLM_API_KEY": "test-key" if llm else None},
            "fake_upstream": {"mode": mode, "delay_s": 1.5, "slow_s": 3.5, "hang_s": 30,
                              "replay": str(REPLAY.relative_to(ROOT))} if fake else None,
            "bundle": h.get("bundle"), "bundle_sha256": h.get("bundle_sha256"), "policy": h.get("policy"),
            "llm": h.get("llm"), "inputs": inputs_meta(ids, path),
            "loadtest": f"scripts/loadtest.py --warmup 0, {PROCS.get(name, 1)} process(es)",
            "load_avg_before": load_before, "grading_note": "graded with the 1,700 unsent dev + dev_noisy rows "
            "excluded: read report.json extras; report_official.json counts unsent rows as wrong",
            "started": datetime.fromtimestamp(time.time() - (time.monotonic() - t_start)).isoformat(timespec="seconds"),
            **machine()}
    write_json(out / "meta.json", meta)
    keys = ("errors", "non_200", "schema_invalid", "latency_ms", "throughput_rps", "llm_status_by_row",
            "breaker_after", "llm_failed_rows", "llm_failed_equal_cpu_only", "stop")
    print(json.dumps({"run": out.name, **{k: summary[k] for k in keys}}))
    return summary


# ----------------------------------------------------------------------------- recovery

def join_status(recs: list[dict], logs: dict) -> None:
    for r in recs:
        line = logs.get(r["request_id"])
        r["llm_status"] = line.get("llm_status") if line else None
        r["logged"] = line is not None


def strip_bodies(recs: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "body"} for r in recs]


def write_preds(out: Path, recs: list[dict]) -> set[str]:
    seen, lines = set(), []
    for r in recs:
        b = r.get("body")
        if r["status"] == 200 and isinstance(b, dict) and b.get("id") not in seen and "-pass" not in str(b.get("id")):
            seen.add(b["id"])
            lines.append(json.dumps(b, ensure_ascii=False) + "\n")
    (out / "predictions.jsonl").write_text("".join(lines), encoding="utf-8")
    return seen


def cmd_recovery(a) -> dict:
    """The upstream answers 500 for the first `switch_s` seconds, then answers well. Traffic: 4 clients at 4
    requests/s each over dev + dev_noisy in order. When does the service use the LLM again?"""
    out = RAW / "chaos-recovery"
    out.mkdir(parents=True, exist_ok=True)
    for f in ("service.log",):
        (out / f).unlink(missing_ok=True)
    fake = Fake("error500", out / "fake_upstream.log")
    svc = Svc(llm_env(f"{fake.url}/v1"), out / "service.log")
    svc.start()
    try:
        start = svc.wait_healthy()
        t0 = time.monotonic()
        poller = Poller(svc.url, 0.25, t0)
        poller.start()
        drv = Driver(svc.url, all_rows(), workers=4, rate=4.0, t0=t0).start()
        time.sleep(a.switch_s)
        fake.set_mode("ok")
        t_switch = time.monotonic() - t0
        time.sleep(a.total_s - a.switch_s)
        recs = drv.stop()
        samples = poller.stop()
        health, metrics = get_json(svc.url + "/healthz"), get_json(svc.url + "/metrics")
        fstate = fake.state()
    finally:
        stop = svc.stop()
        fake.stop()
    logs, other = read_log(out / "service.log")
    join_status(recs, logs)
    llm_recs = [r for r in recs if r["llm_status"]]
    first_ok = next((r for r in llm_recs if r["llm_status"] == "ok" and r["t_start"] >= t_switch), None)
    trans = Poller.transitions(samples)
    opened = next((s["t_s"] for s in samples if s["breaker"] == "open"), None)
    closed_again = next((s["t_s"] for s in samples if opened is not None and s["t_s"] > opened
                         and s["breaker"] == "closed"), None)
    bins = {}
    for r in llm_recs:
        b = int(r["t_start"] // 5) * 5
        bins.setdefault(b, Counter())[r["llm_status"]] += 1
    ids = write_preds(out, recs)
    cpu = read_preds(RAW / "chaos-cpu-ref" / "predictions.jsonl")
    degraded = [r for r in recs if r["llm_status"] in DEGRADED and r["id"] in cpu]
    summary = {"run_id": out.name, "switch_to_ok_at_s": round(t_switch, 3), "total_s": a.total_s,
               "requests": len(recs), "status_counts": dict(Counter(str(r["status"]) for r in recs)),
               "client_errors": dict(Counter(r["error"] for r in recs if r["error"])),
               "llm_status": dict(Counter(r["llm_status"] for r in llm_recs)),
               "breaker_opened_at_s": opened, "breaker_closed_again_at_s": closed_again,
               "first_llm_ok_after_switch_s": first_ok["t_start"] if first_ok else None,
               "seconds_from_upstream_healthy_to_llm_used": round(first_ok["t_start"] - t_switch, 3)
               if first_ok else None,
               "llm_status_per_5s": {str(k): dict(v) for k, v in sorted(bins.items())},
               "breaker_transitions": trans, "fake_upstream": fstate,
               "metrics_after": {k: metrics.get(k) for k in ("requests", "by_status", "llm", "latency_ms")},
               "breaker_after": (health.get("llm") or {}).get("breaker"),
               "degraded_rows_in_cpu_ref": len(degraded),
               "degraded_rows_equal_cpu_only": sum(same(r["body"], cpu[r["id"]]) for r in degraded),
               "start": {k: start[k] for k in ("listening_s", "healthy_s")}, "stop": stop, "log": log_health(other)}
    summary["graded"] = grade(out, ids)
    write_json(out / "summary.json", summary)
    write_json(out / "requests.json", strip_bodies(recs))
    write_json(out / "health_timeline.json", {"every_s": 0.25, "samples": samples})
    h = start["health"]
    write_json(out / "meta.json", {
        "run_id": out.name, "created": now_iso(), "system": "service-http (uvicorn, DESIGN.md serve flags)",
        "changed": f"fake upstream in error500 mode, switched to ok at {round(t_switch, 1)} s; LLM on",
        "traffic": "4 client threads, each at most 4 requests/s, dev then dev_noisy rows in order",
        "bundle": h.get("bundle"), "bundle_sha256": h.get("bundle_sha256"), "policy": h.get("policy"),
        "llm": h.get("llm"), "inputs": {"rows_sent": len(recs), "first_id": recs[0]["id"] if recs else None},
        "grading_note": "graded on the rows sent (others excluded); read report.json extras", **machine()})
    print(json.dumps({k: summary[k] for k in ("switch_to_ok_at_s", "breaker_opened_at_s", "breaker_closed_again_at_s",
                                              "first_llm_ok_after_switch_s", "llm_status", "status_counts")}))
    return summary


# ----------------------------------------------------------------------------- restart and drain

def probe(url: str, row: dict, timeout: float = 30.0) -> dict:
    t = time.monotonic()
    try:
        r = httpx.post(url + "/triage", json=row, timeout=timeout)
        return {"status": r.status_code, "s": round(time.monotonic() - t, 3), "body": r.json()}
    except Exception as e:                                         # noqa: BLE001
        return {"status": None, "error": type(e).__name__, "s": round(time.monotonic() - t, 3)}


def drain_view(recs: list[dict], t_term: float, t_exit: float | None) -> dict:
    """What clients saw around a SIGTERM at t_term."""
    inflight = [r for r in recs if r["t_start"] < t_term <= r["t_end"]]
    after = [r for r in recs if r["t_start"] >= t_term]
    return {"in_flight_at_signal": len(inflight),
            "in_flight_outcomes": dict(Counter(str(r["status"] or r["error"]) for r in inflight)),
            "in_flight_llm_status": dict(Counter(str(r.get("llm_status")) for r in inflight)),
            "in_flight_max_wait_after_signal_s": round(max((r["t_end"] - t_term for r in inflight), default=0), 3),
            "sent_after_signal": len(after),
            "after_signal_outcomes": dict(Counter(str(r["status"] or r["error"]) for r in after)),
            "after_signal_before_exit_200": sum(r["status"] == 200 for r in after
                                                if t_exit is None or r["t_end"] <= t_exit)}


def cmd_restart(a) -> dict:
    out = RAW / "chaos-restart"
    out.mkdir(parents=True, exist_ok=True)
    for f in ("service_a.log", "service_b.log", "service_c.log"):
        (out / f).unlink(missing_ok=True)
    rows = all_rows()
    okref = read_preds(RAW / "chaos-ok" / "predictions.jsonl")
    res: dict = {"run_id": out.name}

    # A: cold start, traffic, kill -9; B: restart on the same port at once, traffic, SIGTERM under traffic
    fake = Fake("ok", out / "fake_upstream.log")
    env = llm_env(f"{fake.url}/v1")
    port = free_port()
    A = Svc(env, out / "service_a.log", port)
    B = Svc(env, out / "service_b.log", port)
    try:
        A.start()
        sa = A.wait_healthy(poll=0.05)
        pa = probe(A.url, rows[0])
        res["cold_start"] = {"listening_s": sa["listening_s"], "healthy_s": sa["healthy_s"],
                             "first_answer_after_healthy_s": pa["s"], "first_answer_status": pa["status"]}
        t0 = time.monotonic()
        poller = Poller(A.url, 0.1, t0)
        poller.start()
        drv = Driver(A.url, rows, workers=4, rate=4.0, t0=t0).start()
        time.sleep(a.before_kill_s)
        t_kill = time.monotonic() - t0
        os.kill(A.p.pid, signal.SIGKILL)
        A.p.wait()
        A.log.close()
        kill_rc = A.p.returncode
        B.start()
        t_respawn = B.t_spawn - t0
        sb = B.wait_healthy(poll=0.05)
        t_healthy = time.monotonic() - t0
        pb = probe(B.url, rows[1])
        t_probe = time.monotonic() - t0
        time.sleep(a.after_restart_s)
        t_term = time.monotonic() - t0
        stopB = B.stop(signal.SIGTERM)
        t_exit = t_term + stopB["exit_s"] if stopB["exit_s"] is not None else None
        time.sleep(1.0)
        recs = drv.stop()
        samples = poller.stop()
    finally:
        for s in (A, B):
            if s.p is not None and s.p.poll() is None:
                s.p.kill()
        fake.stop()
    logs = {**read_log(out / "service_a.log")[0], **read_log(out / "service_b.log")[0]}
    join_status(recs, logs)
    after_kill = [r for r in recs if r["t_start"] >= t_kill and r["t_end"] <= t_term]
    first_good = next((r for r in sorted(after_kill, key=lambda r: r["t_end"]) if r["status"] == 200), None)
    lost = [r for r in recs if r["t_start"] < t_kill < r["t_end"]]
    gap = [r for r in after_kill if r["status"] != 200]
    good = [r for r in recs if r["status"] == 200 and r["id"] in okref]
    res["kill9"] = {
        "killed_at_s": round(t_kill, 3), "kill_exit_code": kill_rc, "respawned_at_s": round(t_respawn, 3),
        "restart_listening_s": sb["listening_s"], "restart_healthy_s": sb["healthy_s"],
        "healthz_polls_during_restart": sb["polls"],
        "kill_to_healthy_s": round(t_healthy - t_kill, 3),
        "probe_after_healthy": {"status": pb["status"], "s": pb["s"]},
        "kill_to_probe_answer_s": round(t_probe - t_kill, 3),
        "traffic_first_200_after_kill_s": round(first_good["t_end"] - t_kill, 3) if first_good else None,
        "in_flight_at_kill": len(lost), "in_flight_at_kill_outcomes": dict(Counter(str(r["status"] or r["error"])
                                                                                    for r in lost)),
        "failed_requests_during_restart": len(gap),
        "failed_during_restart_by_kind": dict(Counter(str(r["error"] or r["error_code"] or r["status"]) for r in gap)),
        "answers_equal_ok_run": f"{sum(same(r['body'], okref[r['id']]) for r in good)} of {len(good)}",
        "answers_before_kill_equal": sum(same(r['body'], okref[r['id']]) for r in good if r["t_end"] < t_kill),
        "answers_after_restart_equal": sum(same(r['body'], okref[r['id']]) for r in good if r["t_start"] > t_kill),
        "answers_after_restart": sum(1 for r in good if r["t_start"] > t_kill)}
    _, otherB = read_log(out / "service_b.log")
    res["sigterm_ok_upstream"] = {"t_signal_s": round(t_term, 3), **stopB, **drain_view(recs, t_term, t_exit),
                                  "log": log_health(otherB), "clean": clean_stop(stopB, log_health(otherB))}
    res["traffic"] = {"requests": len(recs), "status_counts": dict(Counter(str(r["status"]) for r in recs)),
                      "client_errors": dict(Counter(r["error"] for r in recs if r["error"])),
                      "llm_status": dict(Counter(str(r["llm_status"]) for r in recs if r["llm_status"]))}
    write_json(out / "requests_kill9.json", strip_bodies(recs))
    write_json(out / "health_timeline_kill9.json", {"every_s": 0.1, "transitions": Poller.transitions(samples)})
    ids = write_preds(out, recs)

    # C: SIGTERM while LLM calls wait on a hung upstream (each call has a 4 s deadline)
    fake = Fake("hang", out / "fake_upstream_hang.log")
    C = Svc(llm_env(f"{fake.url}/v1"), out / "service_c.log")
    try:
        C.start()
        C.wait_healthy()
        t0 = time.monotonic()
        drv = Driver(C.url, rows, workers=16, rate=100.0, t0=t0).start()
        time.sleep(a.hang_before_term_s)
        waiting = len(drv.outstanding())
        fs = fake.state()
        t_term = time.monotonic() - t0
        stopC = C.stop(signal.SIGTERM, on_signal=drv.halt.set)    # in-flight requests finish; no new ones
        t_exit = t_term + stopC["exit_s"] if stopC["exit_s"] is not None else None
        time.sleep(0.5)
        recsC = drv.stop()
    finally:
        if C.p is not None and C.p.poll() is None:
            C.p.kill()
        fake.stop()
    logsC, otherC = read_log(out / "service_c.log")
    join_status(recsC, logsC)
    res["sigterm_hung_upstream"] = {"t_signal_s": round(t_term, 3), "client_requests_waiting_at_signal": waiting,
                                    "upstream_calls_in_flight_at_signal": fs.get("in_flight"), **stopC,
                                    **drain_view(recsC, t_term, t_exit), "log": log_health(otherC),
                                    "clean": clean_stop(stopC, log_health(otherC))}
    write_json(out / "requests_sigterm_hang.json", strip_bodies(recsC))
    res["graded"] = grade(out, ids)
    write_json(out / "summary.json", res)
    h = sa["health"]
    write_json(out / "meta.json", {
        "run_id": out.name, "created": now_iso(), "system": "service-http (uvicorn, DESIGN.md serve flags)",
        "changed": "process faults: kill -9 under traffic then an immediate restart on the same port (fake upstream "
                   "ok); SIGTERM under traffic (fake upstream ok); SIGTERM while LLM calls hang (fake upstream hang)",
        "traffic": "kill -9 phase: 4 client threads at most 4 requests/s each; hang phase: 16 threads unpaced; "
                   "rows: dev then dev_noisy in order",
        "bundle": h.get("bundle"), "bundle_sha256": h.get("bundle_sha256"), "policy": h.get("policy"),
        "llm": h.get("llm"), "grading_note": "predictions.jsonl holds the kill -9 phase's 200 answers (first per id); "
        "graded with all other rows excluded", **machine()})
    print(json.dumps({k: res[k] for k in ("cold_start", "kill9", "sigterm_ok_upstream", "sigterm_hung_upstream")},
                     default=str)[:3000])
    return res


# ----------------------------------------------------------------------------- report

def cmd_report(_a=None) -> dict:
    runs = {}
    for name in ORDER:
        p = RAW / f"chaos-{name}" / "summary.json"
        if p.exists():
            runs[name] = json.loads(p.read_text())
    extra = {n: json.loads((RAW / f"chaos-{n}" / "summary.json").read_text())
             for n in ("recovery", "restart") if (RAW / f"chaos-{n}" / "summary.json").exists()}
    ref = RAW / "chaos-ref-replay" / "meta.json"
    doc = {"created": now_iso(), "inputs": f"first {N_DEV} dev + first {N_NOISY} dev_noisy rows (600)",
           "ref_replay": json.loads(ref.read_text()) if ref.exists() else None, "load": runs, **extra}
    write_json(HERE / "chaos.json", doc)

    def lat(d):
        return f"{d['p50']:.0f} / {d['p95']:.0f} / {d['p99']:.0f}" if d and d.get("p50") is not None else "-"

    def llm_st(s):
        return ", ".join(f"{k} {v}" for k, v in sorted(s["llm_status_by_row"].items(), key=lambda kv: -kv[1])
                         if k not in ("None", "client_error") and not re.fullmatch(r"http_\d+", k)) or "-"

    def acc(s):
        g = s.get("graded") or {}
        return f"{s['action_acc']:.3f}" if s.get("action_acc") is not None else "-"

    print("Fault modes, 8 clients, 600 rows\n")
    print("| Run | Upstream | Errors / non-200 / schema-invalid | p50 / p95 / p99 ms | LLM status (rows) | "
          "Breaker opened at | Breaker at end | LLM-failed rows equal to CPU-only | Action acc | Routed | SIGTERM stop |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, s in runs.items():
        if s["concurrency"] != 8:
            continue
        eq = (f"{s['llm_failed_equal_cpu_only']} of {s['llm_failed_rows']}" if s["llm_failed_rows"] else "none failed")
        b = s.get("breaker_first_not_closed_s")
        print(f"| chaos-{name} | {s['mode'] or 'LLM off'} | {s['errors']} / {s['non_200']} / {s['schema_invalid']} | "
              f"{lat(s['latency_ms'])} | {llm_st(s)} | {f'{b:.1f} s' if b is not None else '-'} | "
              f"{s['breaker_after'] or '-'} | {eq} | {acc(s)} | {s['routed_share']:.3f} | "
              f"{'clean' if s['log'].get('shutdown_clean') else 'NOT clean'}, {s['stop']['exit_s']} s |")
    print("\nLoad, 600 rows\n")
    print("| Run | Upstream | Clients (processes) | 200 / JSON 503 shed / client errors | Client p50 / p95 / p99 ms | "
          "App p50 / p95 / p99 ms | Probe /metrics p50 / max ms | Load-client CPU / wall s | req/s | "
          "LLM status (rows) | LLM-failed rows equal to CPU-only | Action acc | SIGTERM stop |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, s in runs.items():
        if s["concurrency"] == 8:
            continue
        sc = s["status_counts"]
        procs = PROCS.get(name, 1)
        pr = s.get("probe_rtt_ms", {}).get("metrics_ms") or {}
        eq = (f"{s['llm_failed_equal_cpu_only']} of {s['llm_failed_rows']}" if s["llm_failed_rows"] else "-")
        print(f"| chaos-{name} | {s['mode'] or 'LLM off'} | {s['concurrency']} ({procs}) | {sc.get('200', 0)} / "
              f"{s['error_codes'].get('overloaded', 0)} / {s['errors']} | {lat(s['latency_ms'])} | "
              f"{lat(s['metrics_after'].get('latency_ms'))} | {pr.get('p50', '-')} / {pr.get('max', '-')} | "
              f"{s.get('client_cpu_s', '-')} / {s['wall_s']:.1f} | "
              f"{s['throughput_rps']:.0f} | {llm_st(s)} | {eq} | {acc(s)} | "
              f"{'clean' if s['log'].get('shutdown_clean') else 'NOT clean'} |")
    return doc


def cmd_all(a) -> None:
    cmd_ref_replay()
    for name in ORDER:
        cmd_load(argparse.Namespace(name=name))
    cmd_recovery(argparse.Namespace(switch_s=20.0, total_s=75.0))
    cmd_restart(argparse.Namespace(before_kill_s=10.0, after_restart_s=10.0, hang_before_term_s=1.5))
    cmd_report()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("all")
    sub.add_parser("ref-replay")
    p = sub.add_parser("load")
    p.add_argument("name", choices=ORDER)
    p = sub.add_parser("recovery")
    p.add_argument("--switch-s", type=float, default=20.0)
    p.add_argument("--total-s", type=float, default=75.0)
    p = sub.add_parser("restart")
    p.add_argument("--before-kill-s", type=float, default=10.0)
    p.add_argument("--after-restart-s", type=float, default=10.0)
    p.add_argument("--hang-before-term-s", type=float, default=1.5)
    sub.add_parser("report")
    a = ap.parse_args(argv)
    {"all": cmd_all, "ref-replay": cmd_ref_replay, "load": cmd_load, "recovery": cmd_recovery,
     "restart": cmd_restart, "report": cmd_report}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
