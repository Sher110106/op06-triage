#!/usr/bin/env bash
# OP-06 Triage the Inbox: reproduction entry point. Runs on a plain Linux or macOS box with Python 3.12 and uv.
# No GPU, no network at run time once dependencies are installed; the CPU model needs no API key.
#
#   scripts/reproduce.sh setup        install pinned dependencies into .venv (uv sync)
#   scripts/reproduce.sh verify       check the vendored data against data/SHA256SUMS
#   scripts/reproduce.sh baseline     rerun the published lexical baseline and grade it
#   scripts/reproduce.sh train        train the CPU model (about 2 minutes on an Apple M1)
#   scripts/reproduce.sh eval [RUN]   predict dev + dev_noisy + hostile and grade with the official grader
#   scripts/reproduce.sh all          setup, verify, baseline, train, eval
#   scripts/reproduce.sh serve        start POST /triage on $PORT (Phase 1, not built yet)
#
# Results land in results/raw/<run-id>/: predictions.jsonl, details.jsonl, meta.json,
# report.json (harness summary) and report_official.json (grader.py output, unchanged).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

cmd="${1:-help}"
case "$cmd" in
  setup)
    command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
    uv sync --frozen --no-dev ;;
  verify)
    if command -v sha256sum >/dev/null; then (cd data/op06 && sha256sum -c ../SHA256SUMS)
    else (cd data/op06 && shasum -a 256 -c ../SHA256SUMS); fi ;;
  baseline)
    mkdir -p results/raw/dev-baseline
    $PY -I data/op06/baseline.py --train data/op06/train.jsonl.gz --eval data/op06/dev.jsonl \
      --eval-noisy data/op06/dev_noisy.jsonl --eval-hostile data/op06/hostile.jsonl \
      --pred-out results/raw/dev-baseline/predictions.jsonl
    $PY -m harness.eval grade --pred results/raw/dev-baseline/predictions.jsonl ;;
  train)
    $PY -m triage.classical train ;;
  eval)
    $PY -m harness.eval predict --system classical --run-id "${2:-dev-classical-v0}" --force ;;
  all)
    "$0" setup; "$0" verify; "$0" baseline; "$0" train; "$0" eval ;;
  serve)
    echo "serve: the HTTP service arrives in Phase 1" >&2; exit 2 ;;
  *)
    sed -n '2,16p' "$0" ;;
esac
