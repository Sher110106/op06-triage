# GPU validation extension, 18 October 2026

Sher explicitly authorized using a GPU after the stronger-model comparison finished. This
extension validates the existing neural checkpoint on real CUDA hardware. It does not reopen
the completed confirmation, change its review thresholds, or promote its failed operating guard.

The reachable shared host has one RTX A6000, driver 595.99.02 and advertised CUDA 13.2 support.
At the initial check, another user's vLLM process used about 40.6 GB and the card reported 3,660
MiB free and 100% utilization. Leave existing processes untouched. Use a separate CUDA-enabled
environment with Torch 2.14.1+cu130; preserve the measured CPU environment. Tyrone's configured
Tailscale address timed out. No GPU rental or model API call is required.

## Checks fixed before execution

1. Load the saved `modernbert_joint` checkpoint without changing weights or precision. Validate
   the first 100 selection rows and ten singleton checks against the saved CPU ensemble replay.
   Exact fields are ID, intent, action and review flag; the confidence tolerance stays `1e-5`.
   Use batch size one to bound activation memory. A failure is reported, not hidden by changing
   tolerance or reviewing confirmation labels.
2. Measure single-request inference on those selection inputs with five warmups and GPU work
   synchronized through the output copy. Report shared-host conditions, allocator peak and actual
   device. These timings are not dedicated-GPU capacity or a controlled CPU/GPU speedup claim.
3. Exercise the actual CUDA HTTP service with a bounded selection-only cohort, checking HTTP
   status, schema, echoed IDs and parity against the same frozen responses. Do not overwrite
   canonical CPU or earlier neural benchmark files.
4. In a separate process, attempt eight real optimizer updates on fitting examples only, with
   the existing checkpoint, batch size one and AdamW at `3e-5`. Check finite loss/gradients and
   changed parameters. Save no replacement checkpoint and use no selection/confirmation labels
   for these updates. This is a training feasibility check, not a new quality experiment; its
   step time is not comparable to the earlier batch-16 CPU training.

Cap inference's PyTorch allocator at 3% of the 49,140-MiB card and the training check at 5.5%.
Check current free memory before each process, retaining at least 768 MiB beyond the requested
allocator cap for context/library overhead and other users. If the budget cannot fit, reduce
the bounded probe or use the local Apple GPU as a separately labelled alternative; do not evict
another process or remove its reservation. Abort cleanly on insufficient memory and retain the
failure. Only one of our GPU validation processes runs at a time.

Save commands, environment, resource measurements and successful/failed outputs under
`results/gpu-validation-20261018/`. Publish measured results after the checks. The original
classical-plus-Luna production default remains unchanged unless a separate valid model-quality
experiment later justifies changing it.

## Observed memory repair

The first full-model AdamW attempt reached the fixed allocator cap while materializing a
148-MiB temporary during `optimizer.step()`. Preserve that failed attempt separately. Retry
the same eight fit-only updates using AdamW's fused CUDA implementation (`fused=True`, still
`foreach=False`) to reduce temporary allocation. Keep the original memory cap, batch size,
learning rate and absence of checkpoint/quality selection. This changes optimizer execution
for a mechanics probe, not any frozen comparison result.
