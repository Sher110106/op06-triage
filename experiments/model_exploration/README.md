# Model exploration runbook

This is an optional, reproducible experiment for comparing the current classical reference with a frozen encoder and probability ensemble. It does not change the production model default. Run the commands from the repository root. Generated checkpoints live under the ignored `models/` directory; experiment evidence under `results/` is retained in the private repository.

## Environment

Use Python 3.12, as pinned in the root `pyproject.toml`. The CPU training setup measured for this experiment used 32 threads. Install the normal project and test dependencies first, then the optional neural dependencies:

```bash
uv venv --python 3.12 .venv
uv sync --python .venv/bin/python --group dev
uv pip install --python .venv/bin/python \
  'torch==2.14.1' \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python \
  -r experiments/model_exploration/requirements-neural.txt
```

The requirements file pins Transformers 5.19.0 and the tokenizer/runtime dependencies. Install the CPU PyTorch wheel first, as shown. CUDA is supported by the model scripts when a compatible CUDA-enabled PyTorch build and available device are present, but CUDA has not been tested for this experiment. Do not treat the CPU timing or resource measurements as CUDA results.

Training downloads the model and tokenizer for the exact Hugging Face revision below. Do not substitute a moving branch name when reproducing the run:

```text
answerdotai/ModernBERT-base
8949b909ec900327062f0ebf497f51aef5e6f0c8
```

## Data and decision boundaries

The 20,000 published training rows are divided by conversation into roughly 65% fit, 10% selection, 10% calibration, and 15% confirmation. The partition seeds are fixed in `model_common.py`. Conversations stay within one partition. Fit trains model weights and reference predictors; selection chooses checkpoints and blend weights; calibration fits temperatures, correctness models, and review thresholds; confirmation evaluates the already frozen choice once.

This is a new split of previously studied public training data, not a historically pristine dataset or a hidden test. Calibration measurements are partly optimistic because temperatures and classical calibration use the calibration rows. Selection results guide model development; the current-round confirmation is the held-out check of the complete chosen recipe. Public dev, noisy, and hostile sets are diagnostics after the choice is frozen. Do not use them to choose a model or blend.

## Prepare and fit the reference

For a fresh experiment output directory, record the grouped partitions and fit the classical reference on the matching fit/calibration rows:

```bash
.venv/bin/python -m experiments.model_exploration.model_common
.venv/bin/python -m experiments.model_exploration.evaluate prepare
.venv/bin/python -m experiments.model_exploration.evaluate calibrate-reference
```

`evaluate prepare` writes the partition manifest, refits `reference.joblib`, and exports selection/calibration reference probabilities. `calibrate-reference` fits the reference review thresholds and writes its predictions and gate details. These artifacts are prerequisites for model selection and optional hosted-tail preparation.

## Train the encoder

The measured CPU configuration is 32 threads, three epochs, learning rate `3e-5`, batch size 16, maximum length 512, and seed 20261020. The eight-step feasibility check measured about 1.86 seconds per optimizer step at 32 threads. Pass the 90-minute budget because the command's default is 45 minutes:

```bash
.venv/bin/python -m experiments.model_exploration.neural train \
  --model answerdotai/ModernBERT-base \
  --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 \
  --tag modernbert_joint \
  --device cpu --threads 32 --epochs 3 --lr 3e-5 \
  --bs 16 --maxlen 512 --time-budget-min 90
```

The encoder and intent/flow/action heads train on fit rows. Selection rows choose the checkpoint. The action head receives predicted intent probabilities and encoded observed-action state; gold intent and future actions are targets only. Checkpoint metadata is stored under `models/model-exploration/<tag>/`; training logs are stored under `results/model-exploration-20261017/<tag>/`.

## Optional action-head refinement

The refinement freezes the encoder and the intent and flow heads, caches their pooled fit/selection features, and trains only the action/state branch. It reads fit and selection only. Run it before calibration and confirmation:

```bash
.venv/bin/python -m experiments.model_exploration.refine_head \
  --base-tag modernbert_joint --tag modernbert_head \
  --threads 32 --head-threads 4 --feature-bs 16 \
  --epochs 20 --bs 256 --lr 1e-3 --device cpu
```

Compare `base_action_accuracy` and `best_action_accuracy` in `results/model-exploration-20261017/modernbert_head/head_refinement.json`. Use the refined checkpoint only when its raw selection action accuracy improves by at least 0.005; otherwise keep the base checkpoint. Set `TAG` to the choice and use it consistently below:

```bash
export TAG=modernbert_head
# If the 0.005 rule is not met, use: export TAG=modernbert_joint
```

The smoke test uses a randomly initialized tiny local BERT and synthetic rows; it verifies the actual refinement path without Hub downloads or calibration/confirmation access:

```bash
.venv/bin/python -m experiments.model_exploration.smoke_head
```

## Predict, select, and confirm

Create logits for the selected checkpoint on selection and calibration, then freeze its temperatures, ensemble weights, and review thresholds:

```bash
.venv/bin/python -m experiments.model_exploration.neural predict \
  --tag "$TAG" --splits selection calibration --device cpu --bs 32
.venv/bin/python -m experiments.model_exploration.evaluate select --tag "$TAG"
```

After selection is frozen, produce confirmation logits and run the matched comparison:

```bash
.venv/bin/python -m experiments.model_exploration.neural predict \
  --tag "$TAG" --splits confirmation --device cpu --bs 32
.venv/bin/python -m experiments.model_exploration.evaluate confirm --tag "$TAG"
```

The confirmation command compares the reference, encoder, and selected ensemble on the same rows with the shared gate and policy path. Do not rerun selection or alter the tag, blend, thresholds, or model based on confirmation results. Evaluate public diagnostics separately after freezing:

```bash
.venv/bin/python -m experiments.model_exploration.neural predict \
  --tag "$TAG" --splits dev noisy hostile --device cpu --bs 32
.venv/bin/python -m experiments.model_exploration.evaluate confirm \
  --tag "$TAG" --splits dev noisy hostile
```

Outputs are written below `results/model-exploration-20261017/<tag>/`. CUDA can be selected with `--device cuda` only when available with a compatible Torch build; this experiment has not validated CUDA behavior or performance.

## Verify runtime parity

This compares cached frozen replay against real encoder inference on the first rows of selection. It does not score labels or read calibration, confirmation, dev, noisy, or hostile rows:

```bash
.venv/bin/python -m experiments.model_exploration.verify_runtime \
  --tag "$TAG" --limit 100 --threads 2
```

The report is `results/model-exploration-20261017/<tag>/runtime_parity.json`. A parity failure must be resolved before serving or benchmarking the candidate.

## Optional experimental HTTP service and load test

Start the experimental service only after training, selection/calibration, and runtime parity have produced the artifacts for `TAG`:

```bash
TRIAGE_NEURAL_TAG="$TAG" \
TRIAGE_NEURAL_THREADS=2 \
TRIAGE_NEURAL_DEVICE=cpu \
uvicorn experiments.model_exploration.serve:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

The wrapper reuses the shared Triage request path, installs the calibrated thresholds into the versioned policy snapshot, and reports neural checkpoint provenance through `/healthz`. The policy is frozen at startup and does not hot-reload. The LLM action step and second-intent routing are off to match the controlled comparison. This is an experimental backend; the production `triage.service:app` default remains unchanged.

In another terminal, run the standard development, noisy, and hostile HTTP load test:

```bash
.venv/bin/python scripts/loadtest.py \
  --url http://127.0.0.1:8000 --concurrency 8 --warmup 20 \
  --label "neural-$TAG" \
  --out "results/model-exploration-20261017/$TAG/http-loadtest.json"
```

The load test checks HTTP status, response schema, echoed IDs, latency, and throughput. It does not decide whether to replace the production model. Keep one model-serving process per candidate; each worker loads a separate model copy.

## Optional hosted-model comparison

Hosted calls are a separate, explicit action-only experiment. Neural training, prediction, evaluation, serving, and the tests above do not call a model API. The hosted runner requires all of the following before `run`: a frozen cohort and CPU fallback; `api_models.json` and `api_endpoints.json` with saved model and endpoint prices; `budget_before.json` with a fresh gateway meter snapshot; and a credential file outside the repository. Do not create or edit the price and meter snapshots by guessing values. The global phase cap is $5, with bounded per-run caps and sequential requests. The credential path must not be inside the repository.

Prepare the fixed 300-row selection cohort; this makes no API request:

```bash
.venv/bin/python -m experiments.model_exploration.hosted prepare \
  --split selection --limit 300
```

For each candidate, use the exact model ID from the saved price snapshot and a key file provisioned outside the repository. Example run IDs below are the ones expected by the frozen selection analysis. Set the Luna and Haiku per-run caps to $1.50 and the Flash cap to $2.25; the combined hard phase cap remains $5. Do not use `--limit` for runs that will be analyzed as complete cohorts:

```bash
KEY_FILE=/secure/path/outside/repository/openrouter.key
LUNA_MODEL_ID=openai/gpt-6-luna
HAIKU_MODEL_ID=anthropic/claude-haiku-5.5
FLASH_MODEL_ID=google/gemini-3.8-flash

.venv/bin/python -m experiments.model_exploration.hosted run \
  --key-file "$KEY_FILE" --model "$LUNA_MODEL_ID" \
  --run-id selection-luna \
  --cohort results/model-exploration-20261017/cohorts/selection.json \
  --max-spend-usd 1.5 --concurrency 1

.venv/bin/python -m experiments.model_exploration.hosted run \
  --key-file "$KEY_FILE" --model "$HAIKU_MODEL_ID" \
  --run-id selection-haiku \
  --cohort results/model-exploration-20261017/cohorts/selection.json \
  --max-spend-usd 1.5 --concurrency 1

.venv/bin/python -m experiments.model_exploration.hosted run \
  --key-file "$KEY_FILE" --model "$FLASH_MODEL_ID" \
  --run-id selection-flash-low \
  --cohort results/model-exploration-20261017/cohorts/selection.json \
  --max-spend-usd 2.25 --concurrency 1
```

The Luna, Haiku, and Flash IDs must each match exactly one entry in `api_models.json`; Flash's supported profile uses low reasoning with a 1,024-token output cap and native sampling defaults. The phase cap tracks gateway spend and completed call liabilities, retaining a reservation only for an unresolved started call; finalized calls do not keep their original reservation charged cumulatively. Each request is reserved before sending, finalized once, and not replayed after an interrupted call. Failed, malformed, masked, and abstaining answers retain the frozen CPU fallback.

Analyze complete selection runs offline and freeze the winner:

```bash
.venv/bin/python -m experiments.model_exploration.analyze_hosted select \
  --cohort results/model-exploration-20261017/cohorts/selection.json \
  --runs selection-luna selection-haiku selection-flash-low
```

Only if continuing to the hosted confirmation comparison, prepare a new capped confirmation cohort after selection is frozen. Use the winner and current Luna control on that same cohort, subject to the remaining shared cap; if Luna wins selection, skip a redundant Luna-versus-itself call. Use new run IDs and compare them explicitly:

```bash
.venv/bin/python -m experiments.model_exploration.hosted prepare \
  --split confirmation --limit 300

# Run the selected candidate and the Luna control with distinct new run IDs,
# exact saved model IDs, and reviewed per-run caps before comparing.
.venv/bin/python -m experiments.model_exploration.analyze_hosted compare \
  --cohort results/model-exploration-20261017/cohorts/confirmation.json \
  --runs confirmation-winner confirmation-luna
```

Hosted analysis is offline once the raw attempts are complete. It reports action-only results with failures retained as CPU fallbacks; confidence and review flags are illustrative, not calibrated production decisions. A 4-second replay is retrospective and does not represent provider cancellation or actual service latency.

## Artifacts and interpretation

`models/model-exploration/` stores checkpoints, tokenizer files, the fresh classical reference, and calibration objects; this directory is ignored by Git. `results/model-exploration-20261017/` stores partitions, logits, predictions, frozen settings, comparison reports, and optional hosted ledgers as private experiment evidence. Raw provider request logs are retained privately and must never be copied to the public explainer. Keep credentials out of source files, documentation, result artifacts, and Git; provide them only through a secure path outside the repository when explicitly running the hosted experiment.

This runbook specifies runnable experiments, not a result claim or production change. A model switch requires the frozen confirmation and operational criteria in `PROTOCOL.md`, service integration, schema/noise checks, and measured HTTP latency and memory on the target hardware. CUDA remains untested, and the hosted model branch is never auto-enabled.
