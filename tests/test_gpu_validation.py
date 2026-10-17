"""Host-only safety checks for the isolated CUDA validation harness."""
from __future__ import annotations

import pytest

from experiments.model_exploration import gpu_validate as G


def _gpu(index: int, uuid: str, free: int = 10_000):
    return {"index": index, "uuid": uuid, "name": f"GPU {index}",
            "memory_total_mib": 49_140, "memory_free_mib": free,
            "driver_version": "test-driver"}


def test_visible_cuda_ordinal_maps_to_selected_physical_gpu():
    devices = [_gpu(0, "GPU-a"), _gpu(1, "GPU-b")]

    assert G._select_gpu(devices, "cuda:0", visible="1,0")["uuid"] == "GPU-b"
    assert G._select_gpu(devices, "cuda:1", visible="1,0")["uuid"] == "GPU-a"


def test_capacity_guard_includes_fraction_and_fixed_headroom():
    gpu = _gpu(0, "GPU-a", free=2_243)

    result = G._check_capacity(gpu, 0.03)
    assert result["passes"] is True
    assert result["required_free_mib"] == pytest.approx(2_242.2)


def test_capacity_guard_fails_without_enough_free_memory():
    gpu = _gpu(0, "GPU-a", free=2_242)

    with pytest.raises(RuntimeError, match="GPU memory guard failed"):
        G._check_capacity(gpu, 0.03)


def test_capacity_failure_writes_a_failed_stage_report_without_torch(monkeypatch, tmp_path):
    monkeypatch.setattr(G, "RESULTS", tmp_path)
    monkeypatch.setattr(G, "_query_gpus", lambda: [_gpu(0, "GPU-a", free=100)])

    report = G.run("benchmark")

    saved = __import__("json").loads((tmp_path / "benchmark.json").read_text())
    assert report["pass"] is False
    assert saved["exception"]["type"] == "RuntimeError"
    assert "GPU memory guard failed" in saved["exception"]["message"]
    assert saved["elapsed_seconds"] >= 0
