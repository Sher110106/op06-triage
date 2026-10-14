# OP-06 Triage the Inbox

`POST /triage` takes the last eight turns of a customer-support conversation (ABCD) and returns the
customer's intent, the next action the procedure permits, a confidence, and whether a person should look.

Status: **Phase 0** (skeleton, evaluation harness, reproducible CPU model, experiment log). The HTTP
service arrives in Phase 1. See `EXPERIMENT_LOG.md` for what has been measured and `DECISIONS.md` for why.

## Quick start

```sh
scripts/reproduce.sh all      # install, verify data, rerun the baseline, train, evaluate
```

## Layout

| Path | What |
|---|---|
| `data/op06/` | The published OP-06 files, unchanged, with SHA-256 sums in `data/SHA256SUMS` |
| `triage/` | The model code: text cleaning, dialogue-state parser, the classical CPU model |
| `harness/eval.py` | Runs a system over the projected inputs, calls the official grader, adds bootstrap intervals and paired comparisons |
| `experiments/` | Experiments that inform decisions, each with its own pre-registration or notes |
| `results/raw/<run-id>/` | Predictions, the grader's report and the harness summary for every run |
| `results/manifest.json` | Every run with its seed, model, input hashes and status |
| `scripts/reproduce.sh` | The one entry point the reviewer runs |
