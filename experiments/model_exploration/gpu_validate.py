"""Bounded CUDA validation for the frozen ModernBERT experiment.

Run one stage at a time from the solution root. No training checkpoint is saved and no
provider APIs are called. Reports are isolated from the original model-exploration evidence.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "gpu-validation-20261018"
TAG_DEFAULT = "modernbert_joint"
DEVICE_DEFAULT = "cuda:0"
THREADS_DEFAULT = 4
BATCH_SIZE_DEFAULT = 1
INFERENCE_FRACTION = 0.03
TRAINING_FRACTION = 0.055
FREE_HEADROOM_MIB = 768
PARITY_ROWS = 100
BENCHMARK_ROWS = 100
TRAIN_UPDATES = 8


def _device_index(device: str) -> int:
    match = re.fullmatch(r"cuda(?::(\d+))?", device)
    if not match:
        raise ValueError(f"GPU validation requires a CUDA device, got {device!r}")
    return int(match.group(1) or 0)


def _query_gpus() -> list[dict[str, Any]]:
    command = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,driver_version",
               "--format=csv,noheader,nounits"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
    devices = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        index, uuid, name, total, free, driver = fields
        devices.append({"index": int(index), "uuid": uuid, "name": name,
                        "memory_total_mib": int(total), "memory_free_mib": int(free),
                        "driver_version": driver})
    if not devices:
        raise RuntimeError("nvidia-smi reported no GPUs")
    return devices


def _select_gpu(devices: list[dict[str, Any]], device: str, visible: str | None = None) -> dict[str, Any]:
    logical_index = _device_index(device)
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES") if visible is None else visible
    if visibility:
        tokens = [token.strip() for token in visibility.split(",")]
        if logical_index >= len(tokens):
            raise ValueError(f"{device} is outside CUDA_VISIBLE_DEVICES={visibility!r}")
        token = tokens[logical_index]
        candidates = [gpu for gpu in devices if token in (str(gpu["index"]), gpu["uuid"])]
        if len(candidates) != 1:
            raise ValueError(f"cannot map visible CUDA device {token!r} to one nvidia-smi GPU")
        return candidates[0]
    candidates = [gpu for gpu in devices if gpu["index"] == logical_index]
    if not candidates:
        raise ValueError(f"nvidia-smi has no physical GPU index {logical_index}")
    return candidates[0]


def _check_capacity(gpu: dict[str, Any], fraction: float) -> dict[str, Any]:
    total = gpu["memory_total_mib"]
    free = gpu["memory_free_mib"]
    minimum_free = total * fraction + FREE_HEADROOM_MIB
    result = {"allocator_fraction": fraction, "required_free_mib": round(minimum_free, 2),
              "observed_free_mib": free, "passes": free >= minimum_free}
    if not result["passes"]:
        raise RuntimeError(f"GPU memory guard failed: {free} MiB free; need at least "
                           f"{minimum_free:.2f} MiB for fraction {fraction:.3f} plus "
                           f"{FREE_HEADROOM_MIB} MiB headroom")
    return result


def _gpu_snapshot(device: str) -> list[dict[str, Any]]:
    return _query_gpus()


def _torch_gpu_stats(torch, device: str) -> dict[str, int | None]:
    try:
        return {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}
    except Exception:
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")


def _prepare(stage: str, device: str) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Check host capacity before importing Torch or initializing a CUDA context."""
    devices = _query_gpus()
    selected = _select_gpu(devices, device)
    fraction = TRAINING_FRACTION if stage == "training" else INFERENCE_FRACTION
    guard = _check_capacity(selected, fraction)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Torch reports CUDA unavailable after the nvidia-smi capacity check")
    torch.set_num_threads(THREADS_DEFAULT)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        torch.cuda.synchronize(device)
    except Exception:
        pass
    runtime = {"torch": str(torch.__version__), "torch_cuda": getattr(torch.version, "cuda", None),
               "cuda_available": bool(torch.cuda.is_available()), "device": device,
               "device_name": torch.cuda.get_device_name(device), "threads": THREADS_DEFAULT,
               "allocator_fraction": fraction,
               "memory_headroom_mib": FREE_HEADROOM_MIB}
    return torch, selected, {"selected_gpu": selected, "guard": guard, "runtime": runtime}


def _training_probe(torch, tag: str, device: str) -> dict[str, Any]:
    import numpy as np
    from experiments.model_exploration import model_common as M, neural as N

    seed = 20261026
    N._seed_everything(seed)
    model, tokenizer, metadata, _output, _loaded_device = N._load_checkpoint(tag, device)
    original_mode = model.training
    fit_rows = list(M.partition("fit")[:TRAIN_UPDATES])
    if len(fit_rows) != TRAIN_UPDATES:
        raise RuntimeError(f"fit partition contains only {len(fit_rows)} rows")
    labels = metadata["labels"]
    state_mean = np.asarray(metadata["state_mean"], dtype=np.float32)
    state_std = np.asarray(metadata["state_std"], dtype=np.float32)
    state = N._state_norm(N._state_rows(fit_rows), state_mean, state_std)
    sequences = N.encode_rows(tokenizer, fit_rows, metadata["max_length"])
    y = N._labels_by_row(fit_rows, labels)
    label_tensors = {head: torch.as_tensor(values, dtype=torch.long, device=device)
                     for head, values in y.items()}
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("loaded checkpoint has no trainable parameters")

    named_parameters = dict(model.named_parameters())
    action_probe = named_parameters.get("action.weight")
    encoder_probe = next(((name, parameter) for name, parameter in named_parameters.items()
                          if name.startswith("encoder.") and parameter.ndim == 2
                          and parameter.numel() <= 2_000_000), None)
    if action_probe is None or encoder_probe is None:
        raise RuntimeError("could not identify action and encoder matrix parameters for update verification")
    encoder_probe_name, encoder_probe_parameter = encoder_probe
    action_before = action_probe.detach().cpu().clone()
    encoder_before = encoder_probe_parameter.detach().cpu().clone()
    optimizer = torch.optim.AdamW(trainable, lr=3e-5, weight_decay=0.01, foreach=False, fused=True)
    losses, grad_norms = [], []
    encoder_gradient_norms = []
    started = time.perf_counter()
    try:
        model.train()
        for index in range(TRAIN_UPDATES):
            token_ids, mask = N._tensor_batch(torch, sequences, [index], tokenizer.pad_token_id or 0, device)
            output = model(token_ids, mask, state_tensor[index:index + 1])
            loss = sum(N.LOSS_WEIGHT[head] * torch.nn.functional.cross_entropy(
                output[head], label_tensors[head][index:index + 1]) for head in N.HEADS)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"non-finite loss at update {index + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not any(parameter.grad is not None for parameter in trainable):
                raise FloatingPointError(f"missing or non-finite gradients at update {index + 1}")
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            if not bool(torch.isfinite(norm).item()):
                raise FloatingPointError(f"non-finite gradient norm at update {index + 1}")
            encoder_gradient = encoder_probe_parameter.grad
            if encoder_gradient is None or not bool(torch.isfinite(encoder_gradient).all().item()):
                raise FloatingPointError(f"encoder gradient missing or non-finite at update {index + 1}")
            encoder_gradient_norms.append(float(torch.linalg.vector_norm(encoder_gradient).detach().item()))
            optimizer.step()
            losses.append(float(loss.detach().item()))
            grad_norms.append(float(norm.detach().item()))
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        action_delta = action_probe.detach().cpu() - action_before
        encoder_delta = encoder_probe_parameter.detach().cpu() - encoder_before
        action_max_delta = float(action_delta.abs().max().item())
        encoder_max_delta = float(encoder_delta.abs().max().item())
        changed = action_max_delta > 0 and encoder_max_delta > 0
        if not changed:
            raise RuntimeError("action and encoder probe weights did not both change after eight updates")
        if not all(value > 0 for value in encoder_gradient_norms):
            raise RuntimeError("encoder probe had a zero gradient during training")
        return {"updates": TRAIN_UPDATES, "batch_size": 1, "learning_rate": 3e-5,
                "seed": seed, "parameter_device": str(action_probe.device),
                "trainable_parameters": sum(parameter.numel() for parameter in trainable),
                "optimizer": "AdamW", "foreach": False, "fused": True,
                "fit_row_ids": [row["id"] for row in fit_rows],
                "gradient_updates_used_fit_labels_only": True,
                "selection_or_confirmation_labels_used": False,
                "finite_losses": losses, "finite_gradient_norms": grad_norms,
                "finite_encoder_gradient_norms": encoder_gradient_norms,
                "parameters_changed": changed,
                "changed_parameter_probes": {
                    "action.weight": {"max_abs_delta": action_max_delta},
                    encoder_probe_name: {"max_abs_delta": encoder_max_delta}},
                "checkpoint_saved": False,
                "quality_claim": False, "elapsed_seconds": elapsed,
                "note": "Eight bounded full-model updates verify CUDA training mechanics only; not comparable to batch-16 CPU training."}
    finally:
        model.train(original_mode)
        del optimizer
        del model, tokenizer, trainable, label_tensors, state_tensor
        torch.cuda.empty_cache()


def _benchmark(torch, tag: str, device: str) -> dict[str, Any]:
    from experiments.model_exploration import model_common as M, runtime as R

    legacy_path = Path(M.OUT) / tag / f"runtime_bench_{device.replace(':', '-')}_threads{THREADS_DEFAULT}.json"
    existed = legacy_path.exists()
    prior = legacy_path.read_bytes() if existed else None
    try:
        result = R.benchmark(tag, THREADS_DEFAULT, BENCHMARK_ROWS, device)
        torch.cuda.synchronize(device)
        return result
    finally:
        if existed:
            legacy_path.write_bytes(prior)
        elif legacy_path.exists():
            legacy_path.unlink()


def _serve(torch, tag: str, device: str) -> dict[str, Any]:
    import uvicorn
    from experiments.model_exploration import serve

    os.environ.update({"TRIAGE_NEURAL_TAG": tag, "TRIAGE_NEURAL_DEVICE": device,
                       "TRIAGE_NEURAL_THREADS": str(THREADS_DEFAULT),
                       "TRIAGE_NEURAL_BATCH_SIZE": str(BATCH_SIZE_DEFAULT)})
    config = uvicorn.Config(serve.app, host="127.0.0.1", port=18767, workers=1, log_level="info")
    server = uvicorn.Server(config)
    started = time.time()
    # The private probe shutdown signal lets uvicorn finish normally and preserves
    # the final CUDA report. Its standard SIGTERM handler re-raises on exit.
    stop_signal = getattr(signal, "SIGUSR1", None)
    previous_handler = (signal.signal(stop_signal, lambda *_: setattr(server, "should_exit", True))
                        if stop_signal is not None else None)
    try:
        server.run()
    finally:
        if stop_signal is not None:
            signal.signal(stop_signal, previous_handler)
    try:
        torch.cuda.synchronize(device)
    except Exception:
        pass
    return {"host": "127.0.0.1", "port": 18767, "workers": 1,
            "tag": tag, "device": device, "threads": THREADS_DEFAULT,
            "batch_size": BATCH_SIZE_DEFAULT, "llm_enabled": False,
            "external_http_client_expected": True,
            "probe_shutdown_signal": "SIGUSR1" if stop_signal is not None else None,
            "started_unix": started, "finished_unix": time.time(),
            "server_should_exit": server.should_exit}


def run(stage: str, tag: str = TAG_DEFAULT, device: str = DEVICE_DEFAULT) -> dict[str, Any]:
    if stage not in {"parity", "benchmark", "training", "serve"}:
        raise ValueError(f"unknown stage {stage!r}")
    report_path = RESULTS / f"{stage}.json"
    report: dict[str, Any] = {"stage": stage, "tag": tag, "requested_device": device,
                              "started_unix": time.time(), "pass": False,
                              "scope": "bounded CUDA operational validation; no model quality claim"}
    started = time.perf_counter()
    torch = None
    try:
        # Capture the host state before _prepare imports Torch or touches CUDA.
        report["gpu_before"] = _query_gpus()
        torch, selected, prepared = _prepare(stage, device)
        report.update(prepared)
        _write_report(report_path, report)
        if stage == "parity":
            from experiments.model_exploration.verify_runtime import verify
            result = verify(tag, PARITY_ROWS, THREADS_DEFAULT, device=device, batch_size=1)
            report["parity"] = result
            if not result.get("pass"):
                raise RuntimeError("CUDA parity validation reported failure")
        elif stage == "benchmark":
            report["benchmark"] = _benchmark(torch, tag, device)
        elif stage == "training":
            report["training_probe"] = _training_probe(torch, tag, device)
        else:
            report["serve"] = _serve(torch, tag, device)
        report["pass"] = True
    except BaseException as exc:
        report["exception"] = {"type": type(exc).__name__, "message": str(exc),
                               "is_cuda_oom": bool(torch is not None and isinstance(
                                   exc, getattr(torch.cuda, "OutOfMemoryError", ())))}
        report["traceback"] = traceback.format_exc()
    finally:
        if torch is not None:
            report["cuda_peak"] = _torch_gpu_stats(torch, device)
        try:
            report["gpu_after"] = _query_gpus()
        except Exception as exc:
            report["gpu_after_error"] = f"{type(exc).__name__}: {exc}"
        report["finished_unix"] = time.time()
        report["elapsed_seconds"] = time.perf_counter() - started
        _write_report(report_path, report)
    print(json.dumps({"report": str(report_path), **report}, indent=2, default=str), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("parity", "benchmark", "training", "serve"))
    parser.add_argument("--tag", default=TAG_DEFAULT)
    parser.add_argument("--device", default=DEVICE_DEFAULT)
    args = parser.parse_args()
    report = run(args.stage, args.tag, args.device)
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
