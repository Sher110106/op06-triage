# OP-06 Triage the Inbox

`POST /triage` takes the last eight turns of a customer-support conversation (ABCD) and returns the
customer's intent, the next action the procedure permits, a confidence, and whether a person should look.

Status: **Phase 0** (skeleton, evaluation harness, reproducible CPU model, experiment log). The HTTP
service arrives in Phase 1. See `EXPERIMENT_LOG.md` for what has been measured and `DECISIONS.md` for why.

## Quick start

```sh
scripts/reproduce.sh setup        # install the locked dependencies with uv
scripts/reproduce.sh serve        # POST /triage on $PORT (default 8000); trains first if models/ is empty
scripts/reproduce.sh all          # setup, verify data, baseline, train, test, evaluate, smoke test
scripts/reproduce.sh http-eval    # all 2,309 rows through the running service, graded; writes results/*.jsonl
scripts/reproduce.sh help         # every subcommand
```

```sh
curl -s localhost:8000/triage -H 'content-type: application/json' \
  -d '{"id": "x-1", "context": [{"speaker": "customer", "text": "where is my order?"}]}'
```

The service needs no API key. Without one the LLM step is off and the CPU model answers every request; set
`TRIAGE_LLM_API_KEY` (or `OPENROUTER_API_KEY`) to turn it on. `docs/DESIGN.md` lists every setting. Before
changing `policy/policy.toml`, run `scripts/replay_policy.py --policy NEW.toml` to see what the change does.

## Layout

| Path | What |
|---|---|
| `data/op06/` | The published OP-06 files, unchanged, with SHA-256 sums in `data/SHA256SUMS` |
| `triage/` | The service: `gate.py` (input checks), `classical.py` (CPU model), `policy.py`, the LLM step (`prompts.py`, `retrieval.py`, `llm_action.py`), `pipeline.py` (the request path) and `service.py` (HTTP) |
| `policy/policy.toml` | Permitted actions per flow, risk tiers, routing points and rules; the service reloads it on save |
| `models/` | Built by `scripts/reproduce.sh train`, not committed |
| `harness/eval.py` | Runs a system over the projected inputs, calls the official grader, adds bootstrap intervals and paired comparisons |
| `scripts/` | `reproduce.sh` (the one entry point), `loadtest.py`, `replay_policy.py`, and the model and policy builders |
| `tests/` | Unit tests per module; `test_e2e.py` runs the real service over HTTP |
| `docs/DESIGN.md` | Module boundaries, interfaces and settings |
| `experiments/` | Experiments that inform decisions, each with its own pre-registration or notes |
| `results/*.jsonl` | Predictions returned by the running service over HTTP, from `scripts/reproduce.sh http-eval` |
| `results/raw/<run-id>/` | Predictions, the grader's report and the harness summary for every run |
| `results/manifest.json` | Every run with its seed, model, input hashes and status |
