# CUDA validation results, 18 October 2026

The saved `modernbert_joint` inference checkpoint ran on a shared NVIDIA RTX A6000 with CUDA. This was an operational check of the frozen model and HTTP service, plus an eight-step training mechanics probe. It did not change the checkpoint, revisit confirmation, or alter the current CPU-plus-Luna default. The earlier high-risk review guard still fails for the neural candidate, so these results do not justify promoting it.

## Results

| Check | Measurement | Outcome |
|---|---|---|
| Checkpoint parity | 100 selection rows in batch mode and 10 singleton requests; CUDA FP32, batch 1, 4 Torch threads | Exact IDs, intent/action labels, and review flags matched the saved CPU replay. Maximum confidence difference was `2e-6`, within the unchanged `1e-5` tolerance. |
| Predictor benchmark | 100 selection rows after 5 warmups; includes customer-intent probe, policy and confidence work | p50 61.41 ms; p95 65.85 ms. Main prediction alone was p50 33.70 ms / p95 36.42 ms. |
| HTTP load test | 100 saved selection inputs, concurrency 8, 5 warmups | 0 errors, non-200s, invalid schemas, ID mismatches, or replays. Exact output parity held within `2e-6`. p50 552.466 ms; p95 570.640 ms; p99 577.141 ms; max 577.153 ms; throughput 14.45 requests/s. |
| Training mechanics | First 8 FIT rows only; 8 full-model updates, batch 1, AdamW, learning rate `3e-5` | All losses and global gradient norms were finite; encoder gradients were nonzero; sampled action and encoder weights changed. No checkpoint was saved and no quality score was produced. |

The CUDA parity check did not score labels. The HTTP rows came from the frozen selection partition; no development, noisy, hostile, calibration, or confirmation rows were used. The matched CPU expected outputs and request cohort are kept next to the reports.

## Memory and shared-host conditions

The card reports 49,140 MiB total memory. Another user's process was left untouched. `nvidia-smi` snapshots showed 100% utilization at the start and after the checks; utilization was not continuously sampled. Inference used a 3% PyTorch allocator cap plus a 768 MiB free-memory guard. Training used a 5.5% cap with the same guard. A pre-run snapshot had about 3.6 GiB free; snapshots varied as the other workload ran.

For the benchmark and HTTP service, the PyTorch peak was 634,838,016 bytes allocated (about 605.43 MiB) and 643,825,664 bytes reserved (614 MiB). The parity stage peaked slightly higher at 634,893,824 bytes allocated and 645,922,816 bytes reserved. For the successful training probe, peak allocation was 2,677,527,552 bytes (about 2.49 GiB) and reservation was 2,724,200,448 bytes (about 2.54 GiB). These are PyTorch allocator peaks, not total card use or process RSS. After the server was stopped, the port was closed and a later host check showed 3,624 MiB free.

The first training attempt using unfused AdamW hit the fixed cap when an optimizer temporary requested another 148 MiB. That failure is preserved in `training_adamw_unfused_oom.json` and its log. The bounded retry kept the same cap, fit rows, batch size, learning rate, and number of updates, while selecting AdamW's fused CUDA implementation with `foreach=False`. It completed eight updates over 149,841,224 trainable parameters. The update loop took 1.122 seconds and the whole process took 15.304 seconds. This is only evidence that the update path runs; batch-1 CUDA mechanics are not comparable to the earlier batch-16 CPU training run.

## Cost and environment

No GPU rental or model API call was made, and there was no billed cloud expense. The host was a shared RTX A6000, driver `595.99.02`; the isolated validation environment used Python 3.12.15, PyTorch `2.14.1+cu130`, and CUDA runtime `13.0`. The measured CPU environment was left intact. The initial tensor smoke/setup time was not metered. The six recorded GPU stage wall times total 107.178 seconds (0.029772 hours), which normalizes to about `$0.01757` at the [RunPod Secure Cloud RTX A6000 rate](https://www.runpod.io/pricing) of `$0.59/hour`, checked 18 October 2026. The time and price calculation are recorded in [`compute_cost.json`](../../results/gpu-validation-20261018/compute_cost.json). That is a comparison estimate, not a charge or measured GPU-busy time. To account for unmetered setup and idle time, the conservative extension allowance is one whole GPU-hour, or `$0.59` at that rate. No GPU rental was used.

## Reproduction

Audit the saved reports without launching another process when possible. For a fresh live check,
use a separate working copy or archive `results/gpu-validation-20261018/` first: stage commands
write fixed result filenames. Preserve the original failed attempt and measured outputs.

First download and verify the private `modernbert-inference-20261018.tar` asset as described in the [experiment runbook](README.md#use-the-saved-evidence-and-checkpoint). Keep the CUDA environment separate from the existing CPU environment:

```bash
uv venv --python 3.12 .venv-gpu
UV_PROJECT_ENVIRONMENT=.venv-gpu uv sync --python 3.12 --group dev
uv pip install --python .venv-gpu/bin/python 'torch==2.14.1' \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv-gpu/bin/python \
  -r experiments/model_exploration/requirements-neural.txt
```

The recorded runtime was PyTorch `2.14.1+cu130` / CUDA `13.0`; the host driver reported `595.99.02`. Confirm that the selected GPU has the required free memory before running a stage. The validation command checks the GPU with `nvidia-smi` before importing Torch, applies the fixed allocator cap, and saves each stage under `results/gpu-validation-20261018/`:

```bash
.venv-gpu/bin/python -m experiments.model_exploration.gpu_validate --stage parity
.venv-gpu/bin/python -m experiments.model_exploration.gpu_validate --stage benchmark
.venv-gpu/bin/python -m experiments.model_exploration.gpu_validate --stage training
```

The harness defaults to `modernbert_joint`, `cuda:0`, 4 Torch threads, and batch size 1. It refuses to continue when its free-memory guard fails. The fused training probe is the current bounded path. The failed unfused attempt above is retained for audit; do not treat its OOM as a reason to increase the cap or displace another process.

To repeat the actual HTTP run, prepare the same 100 selection request rows and frozen CPU expected predictions, then start the service in one terminal and load-test from another. The saved files `http_inputs.jsonl`, `http_expected.jsonl`, `http_predictions.jsonl`, `http_parity.json`, and `http_loadtest.json` record the measured cohort and results.

```bash
.venv-gpu/bin/python -m experiments.model_exploration.gpu_validate --stage serve &
GPU_PID=$!
.venv-gpu/bin/python scripts/loadtest.py \
  --url http://127.0.0.1:18767 --concurrency 8 --warmup 5 \
  --wait-s 60 --timeout-s 30 \
  --inputs results/gpu-validation-20261018/http_inputs.jsonl --limit 100 \
  --label cuda-selection-100 \
  --out results/gpu-validation-20261018/http_loadtest.json \
  --pred-out results/gpu-validation-20261018/http_predictions.jsonl
kill -USR1 "$GPU_PID"
wait "$GPU_PID"
```

The `SIGUSR1` stop lets the service stage shut down cleanly and write its final allocator and device snapshot. Compare the response file by ID against the saved expected JSONL using exact ID/intent/action/review fields and the unchanged confidence tolerance. The service stage does not itself call an external provider.

The focused harness suite passed 16 tests in the local CPU-only environment (`targeted-tests.log`); this test run itself did not use CUDA. The real CUDA results above are recorded separately by stage. This is not a full-suite run or the public grader's full development/noisy/hostile bar. The GPU and CPU measurements came from different software, hardware and contention conditions, so they do not establish a controlled speedup.

For the predeclared scope and safeguards, see [GPU_VALIDATION.md](GPU_VALIDATION.md). The original CPU evidence and default remain intact.
