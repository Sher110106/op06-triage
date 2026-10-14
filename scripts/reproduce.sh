#!/usr/bin/env bash
# OP-06 Triage the Inbox: the one entry point. Runs on Linux or macOS with bash, Python 3.12 and uv.
# No GPU. The service needs no API key: without one the LLM step is off and the CPU model answers.
#
#   scripts/reproduce.sh setup [--offline]  install the locked dependencies into .venv (uv sync --frozen);
#                                           --offline installs from the local uv cache only
#   scripts/reproduce.sh verify             check data/op06 against data/SHA256SUMS
#   scripts/reproduce.sh test [ARGS]        run the tests (pytest -q; API keys removed from the environment)
#   scripts/reproduce.sh baseline           rerun the published lexical baseline and grade it
#   scripts/reproduce.sh train              train the CPU model (about 2 min on an Apple M1), build the retrieval
#                                           index for the LLM step (about 15 s), and check that
#                                           policy/policy.toml still matches the published permitted lists
#   scripts/reproduce.sh eval [RUN]         run the pipeline in-process over dev + dev_noisy + hostile, LLM step
#                                           as the environment says, grade with the official grader, and
#                                           rebuild results/manifest.json
#   scripts/reproduce.sh serve              start the service on $PORT (default 8000) in the foreground;
#                                           this is the service start. Ctrl-C or SIGTERM stops it.
#   scripts/reproduce.sh smoke              start the service in the background, send the 9 hostile rows,
#                                           20 dev rows and malformed requests, check every answer, stop it
#   scripts/reproduce.sh http-eval [RUN]    start the service, send all 2,309 rows over HTTP at concurrency 8,
#                                           write results/{dev,noisy}_predictions.jsonl, hostile_results.jsonl,
#                                           predictions.jsonl and latency_bench.json, grade, stop the service,
#                                           rebuild results/manifest.json
#   scripts/reproduce.sh loadtest [ARGS]    load-test a service already running at $URL (scripts/loadtest.py)
#   scripts/reproduce.sh all                setup, verify, baseline, train, test, eval, smoke; never blocks
#                                           (http-eval is separate: it rewrites the committed results/*.jsonl)
#   scripts/reproduce.sh help               print this text
#
# With no subcommand the script starts the service, as the OP-06 contract asks (same as serve).
# serve, eval, smoke and http-eval train the CPU model first if models/classical_v0.joblib is missing, and
# build models/retrieval.joblib first if the LLM step is on and the index is missing.
# Default RUN: dev-pipeline-cpu or dev-pipeline-llm (eval), dev-http-cpu or dev-http-llm (http-eval).
# Each run writes results/raw/<RUN>/: predictions.jsonl, report.json (harness summary) and report_official.json
# (grader.py output, unchanged); eval adds details.jsonl and meta.json, http-eval adds meta.json and
# latency_bench.json.
#
# Environment: PORT (8000), HOST (0.0.0.0, serve only), URL (loadtest), WAIT_S (seconds to wait for /healthz,
# 180), and the TRIAGE_* settings in docs/DESIGN.md. The LLM step is on when TRIAGE_LLM=on, or when
# TRIAGE_LLM=auto (the default) and TRIAGE_LLM_API_KEY or OPENROUTER_API_KEY is set.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
BUNDLE=models/classical_v0.joblib
RETRIEVAL=models/retrieval.joblib
# The run command in docs/DESIGN.md. uvicorn counts open connections, idle keep-alive ones included, so 64
# turned away most of 100 clients that connect at once (plain-text 503); 256 serves them all, and the app sheds
# load itself past 128 requests in flight (JSON 503 with Retry-After). --no-access-log keeps stdout to the
# service's one JSON line per request.
SERVE_ARGS=(--workers 1 --limit-concurrency 256 --timeout-keep-alive 5 --no-access-log)
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"     # local calls never go through a proxy

say() { echo "reproduce: $*" >&2; }
die() { echo "reproduce: $*" >&2; exit 1; }

TMP=""
SERVER_PID=""
cleanup() {
  stop_service
  if [ -n "$TMP" ]; then rm -rf "$TMP"; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

tmpdir() { [ -n "$TMP" ] || TMP=$(mktemp -d "${TMPDIR:-/tmp}/op06.XXXXXX"); }

need_venv() { [ -x "$PY" ] || cmd_setup; }

# One line on stderr saying whether the LLM step is on. Sets LLM_ON to 1 or 0; stops on a bad TRIAGE_* value or
# a policy file that will not load, before any server starts.
llm_check() {
  local rc=0 line
  line=$("$PY" - <<'EOF'
import os, sys
from urllib.parse import urlsplit
from triage.llm_action import LLMConfig
from triage.pipeline import Settings
from triage.policy import DEFAULT_POLICY, load
try:
    c = LLMConfig.from_env()
    Settings.from_env()
    load(os.environ.get("TRIAGE_POLICY_PATH") or DEFAULT_POLICY)
except ValueError as e:                 # PolicyError is a ValueError
    print(f"bad setting: {e}")
    sys.exit(2)
if c.enabled:
    print(f"LLM step: on ({c.model} at {urlsplit(c.base_url).hostname}, asked when CPU confidence < {c.gate}, "
          f"timeout {c.timeout_s} s)")
    sys.exit(0)
why = "TRIAGE_LLM=off" if c.mode == "off" else "no API key set"
print(f"LLM step: off ({why}); the CPU model answers every request")
sys.exit(1)
EOF
  ) || rc=$?
  [ "$rc" -ne 2 ] || die "$line"
  say "$line"
  if [ "$rc" -eq 0 ]; then LLM_ON=1; else LLM_ON=0; fi
}

# Train what is missing: the CPU model always, the retrieval index only when the LLM step is on ($1 = 1).
ensure_models() {
  local bundle="${TRIAGE_BUNDLE:-$BUNDLE}"
  if [ ! -f "$bundle" ]; then
    [ -z "${TRIAGE_BUNDLE:-}" ] || die "TRIAGE_BUNDLE=$bundle does not exist"
    say "$BUNDLE is missing; training it first (about 2 min)"
    "$PY" -m triage.classical train
  fi
  if [ "${1:-0}" = 1 ] && [ ! -f "$RETRIEVAL" ]; then
    say "$RETRIEVAL is missing and the LLM step is on; building it first (about 15 s)"
    "$PY" scripts/build_retrieval.py
  fi
}

free_port() {
  "$PY" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'
}

# Exit status 0 when GET /healthz is 200, 2 when the service says the model failed to load, else 1.
healthy() {
  "$PY" -I - "$1" <<'EOF' 2>/dev/null
import json, sys, urllib.error, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    sys.exit(0 if opener.open(sys.argv[1] + "/healthz", timeout=2).status == 200 else 1)
except urllib.error.HTTPError as e:
    sys.exit(2 if e.code == 503 and json.loads(e.read() or b"{}").get("status") == "error" else 1)
except Exception:
    sys.exit(1)
EOF
}

# Start the service in the background on a free local port and wait for /healthz 200. Sets URL and SERVER_PID.
start_service() {
  need_venv
  llm_check
  ensure_models "$LLM_ON"
  tmpdir
  local port wait_s="${WAIT_S:-180}"
  port=$(free_port)
  SERVER_LOG="$TMP/server.log"
  "$PY" -m uvicorn triage.service:app --host 127.0.0.1 --port "$port" "${SERVE_ARGS[@]}" >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  URL="http://127.0.0.1:$port"
  local deadline=$((SECONDS + wait_s)) rc
  while true; do
    rc=0
    healthy "$URL" || rc=$?
    [ "$rc" -ne 0 ] || break
    if [ "$rc" -eq 2 ]; then
      tail -n 30 "$SERVER_LOG" >&2
      die "the service failed to load (GET /healthz says error; see the log above)"
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -n 30 "$SERVER_LOG" >&2
      SERVER_PID=""
      die "the service stopped during start-up"
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      tail -n 30 "$SERVER_LOG" >&2
      die "no 200 from $URL/healthz after $wait_s s"
    fi
    sleep 0.5
  done
  say "service ready at $URL (pid $SERVER_PID)"
}

stop_service() {
  [ -n "$SERVER_PID" ] || return 0
  local pid="$SERVER_PID" i
  SERVER_PID=""
  kill -TERM "$pid" 2>/dev/null || return 0
  for i in $(seq 1 50); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.2
  done
  kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

# ----------------------------------------------------------------------------- subcommands

cmd_setup() {
  command -v uv >/dev/null || die "uv is required: https://docs.astral.sh/uv/getting-started/installation/"
  # --inexact: leave extra packages alone (the optional LLM-experiment group, for one)
  if [ "${1:-}" = "--offline" ]; then
    uv sync --frozen --inexact --offline
  elif ! uv sync --frozen --inexact; then
    [ -d "$(uv cache dir 2>/dev/null)" ] || die "uv sync failed and there is no uv cache to install from"
    say "uv sync failed; trying again from the local uv cache only"
    uv sync --frozen --inexact --offline
  fi
}

cmd_verify() {
  if command -v sha256sum >/dev/null; then (cd data/op06 && sha256sum -c ../SHA256SUMS)
  else (cd data/op06 && shasum -a 256 -c ../SHA256SUMS); fi
}

cmd_test() {
  need_venv
  env -u OPENROUTER_API_KEY -u TRIAGE_LLM_API_KEY "$PY" -m pytest -q "$@"
}

cmd_baseline() {
  need_venv
  mkdir -p results/raw/dev-baseline
  "$PY" -I data/op06/baseline.py --train data/op06/train.jsonl.gz --eval data/op06/dev.jsonl \
    --eval-noisy data/op06/dev_noisy.jsonl --eval-hostile data/op06/hostile.jsonl \
    --pred-out results/raw/dev-baseline/predictions.jsonl
  "$PY" -m harness.eval grade --pred results/raw/dev-baseline/predictions.jsonl
}

# policy/policy.toml may carry the manager's edits (rules, thresholds, version); its flows and alias must
# still equal a fresh build from the published lists.
check_policy() {
  tmpdir
  "$PY" scripts/build_policy.py --out "$TMP/policy.toml" >/dev/null
  "$PY" - "$TMP/policy.toml" "${TRIAGE_POLICY_PATH:-policy/policy.toml}" <<'EOF'
import sys
from pathlib import Path
from triage.policy import load
fresh, path = Path(sys.argv[1]), Path(sys.argv[2])
a, b = load(fresh), load(path)
bad = sorted(f for f in a.flow_permitted if a.flow_permitted[f] != b.flow_permitted.get(f))
if bad or a.alias != b.alias:
    sys.exit(f"{path}: flows or alias differ from the published lists ({', '.join(bad) or 'alias'}); "
             "run scripts/build_policy.py to rebuild it")
n = len(b.rules)
how = "identical to a fresh build" if fresh.read_bytes() == path.read_bytes() else \
    f"edited (version {b.version}, {n} rule{'' if n == 1 else 's'})"
print(f"{path}: flows and alias match the published lists; {how}")
EOF
}

cmd_train() {
  need_venv
  "$PY" -m triage.classical train
  "$PY" scripts/build_retrieval.py
  check_policy
}

cmd_eval() {
  need_venv
  llm_check
  ensure_models "$LLM_ON"
  local run="${1:-}"
  if [ -z "$run" ]; then
    if [ "$LLM_ON" = 1 ]; then run=dev-pipeline-llm; else run=dev-pipeline-cpu; fi
  fi
  "$PY" -m harness.eval predict --system pipeline --run-id "$run" --force
  manifest
}

# Rebuild results/manifest.json from every run on disk; a failure here does not undo the run.
manifest() {
  "$PY" -m harness.manifest >/dev/null || say "could not rebuild results/manifest.json"
}

cmd_serve() {
  need_venv
  llm_check
  ensure_models "$LLM_ON"
  say "starting on http://${HOST:-0.0.0.0}:${PORT:-8000}; GET /healthz returns 200 once the model is loaded"
  exec "$PY" -m uvicorn triage.service:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" "${SERVE_ARGS[@]}"
}

cmd_smoke() {
  start_service
  "$PY" - "$URL" <<'EOF'
import importlib.util, json, sys, time
import httpx
from triage.data import DATA, load_split, project

spec = importlib.util.spec_from_file_location("op06_grader", DATA / "grader.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
hostile = [project(r) for r in load_split("hostile")]
dev = [project(r) for r in load_split("dev")[:20]]
fails, slowest = [], 0.0
with httpx.Client(base_url=sys.argv[1], timeout=10, trust_env=False) as c:
    for row in hostile + dev:
        t = time.perf_counter()
        r = c.post("/triage", content=json.dumps(row).encode("ascii"), headers={"content-type": "application/json"})
        slowest = max(slowest, time.perf_counter() - t)
        body = r.json() if r.status_code == 200 else None
        ok = body is not None and G.schema_valid(body, None) and body.get("id") == row["id"]
        if row in hostile:
            ok = ok and body["needs_human"] is True
        if not ok:
            fails.append(f"{row['id']}: HTTP {r.status_code} {r.text[:200]}")
    big = b'{"id": "big", "context": [{"speaker": "customer", "text": "' + b"a" * (2 << 20) + b'"}]}'
    probes = [("bad JSON", b'{"id": "x", "context": [', 400), ("not an object", b"[1, 2]", 400),
              ("missing id", b'{"context": [{"speaker": "customer", "text": "hi"}]}', 400),
              ("id not text", b'{"id": 7, "context": []}', 400), ("2 MiB body", big, 413)]
    for name, raw, want in probes:
        r = c.post("/triage", content=raw, headers={"content-type": "application/json"})
        try:
            err = r.json().get("error")
        except ValueError:
            err = None
        if r.status_code != want or not (isinstance(err, dict) and err.get("code")):
            fails.append(f"probe {name}: HTTP {r.status_code}, wanted {want}: {r.text[:200]}")
    if c.get("/healthz").status_code != 200:
        fails.append("GET /healthz is not 200 after the probes")
    m = c.get("/metrics").json()
print(json.dumps({"rows": len(hostile) + len(dev), "probes": len(probes), "failures": len(fails),
                  "slowest_s": round(slowest, 3), "by_status": m["by_status"], "llm_calls": m["llm"]["calls"]}))
for f in fails:
    print("FAIL", f, file=sys.stderr)
sys.exit(1 if fails else 0)
EOF
  stop_service
  say "smoke passed"
}

cmd_http_eval() {
  local run="${1:-}"
  start_service
  if [ -z "$run" ]; then
    if [ "$LLM_ON" = 1 ]; then run=dev-http-llm; else run=dev-http-cpu; fi
  fi
  local out="results/raw/$run"
  mkdir -p "$out"
  "$PY" scripts/loadtest.py --url "$URL" --concurrency 8 --wait-s 10 --label "$run" \
    --out results/latency_bench.json --pred-out "$TMP/predictions.jsonl"
  stop_service
  cp results/latency_bench.json "$out/latency_bench.json"
  "$PY" - "$TMP/predictions.jsonl" "$out" <<'EOF'
import json, platform, sys, time
from pathlib import Path
from harness.eval import sha256
from triage.data import DATA, SPLITS, load_split

preds = {p["id"]: p for p in map(json.loads, Path(sys.argv[1]).read_text(encoding="utf-8").splitlines())}
out = Path(sys.argv[2])
files = {"dev": "results/dev_predictions.jsonl", "noisy": "results/noisy_predictions.jsonl",
         "hostile": "results/hostile_results.jsonl"}
every = []
for split, path in files.items():
    lines = [json.dumps(preds[r["id"]], ensure_ascii=False) + "\n" for r in load_split(split) if r["id"] in preds]
    Path(path).write_text("".join(lines), encoding="utf-8")
    every += lines
Path("results/predictions.jsonl").write_text("".join(every), encoding="utf-8")
(out / "predictions.jsonl").write_text("".join(every), encoding="utf-8")
bench = json.loads((out / "latency_bench.json").read_text(encoding="utf-8"))
h = bench["server_health"]
meta = {"run_id": out.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "system": "service-http",
        "bundle": h.get("bundle"), "bundle_sha256": h.get("bundle_sha256"), "policy": h.get("policy"),
        "llm": h.get("llm"), "inputs": {k: {"file": f"data/op06/{SPLITS[k]}", "sha256": sha256(DATA / SPLITS[k])}
                                        for k in files},
        "rows": len(every), "concurrency": bench["concurrency"], "wall_s": bench["wall_s"],
        "latency_ms": bench["latency_ms"], "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()}",
        "note": "every prediction came back over HTTP from the running service (scripts/loadtest.py)"}
(out / "meta.json").write_text(json.dumps(meta, indent=1))
print(f"wrote {len(every)} predictions to results/predictions.jsonl and the three split files")
EOF
  "$PY" -m harness.eval grade --pred results/predictions.jsonl --out "$out/report.json"
  manifest
}

cmd_loadtest() {
  need_venv
  "$PY" scripts/loadtest.py --url "${URL:-http://127.0.0.1:${PORT:-8000}}" "$@"
}

cmd_all() {
  cmd_setup
  cmd_verify
  cmd_baseline
  cmd_train
  cmd_test
  cmd_eval
  cmd_smoke
}

cmd="${1:-serve}"
[ "$#" -eq 0 ] || shift
case "$cmd" in
  setup) cmd_setup "$@" ;;
  verify) cmd_verify ;;
  test) cmd_test "$@" ;;
  baseline) cmd_baseline ;;
  train) cmd_train ;;
  eval) cmd_eval "$@" ;;
  serve) cmd_serve ;;
  smoke) cmd_smoke ;;
  http-eval) cmd_http_eval "$@" ;;
  loadtest) cmd_loadtest "$@" ;;
  all) cmd_all ;;
  help | -h | --help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//' ;;
  *) echo "unknown subcommand: $cmd (try: scripts/reproduce.sh help)" >&2; exit 2 ;;
esac
