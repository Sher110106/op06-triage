"""Device plumbing checks for the optional runtime parity command."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from experiments.model_exploration import verify_runtime as V


class FakeTorch:
    def __init__(self, cuda_available: bool):
        self.cuda = SimpleNamespace(is_available=lambda: cuda_available)

    @staticmethod
    def get_num_threads():
        return 2


class FakePredictor:
    def __init__(self, tag, *, threads, device, batch_size):
        self.received = (tag, threads, device, batch_size)
        self.metadata = {"device": device, "batch_size": batch_size}
        self.closed = False

    def close(self):
        self.closed = True


class FakeRunner:
    def __init__(self, predictor):
        self.predictor = predictor
        self.llm = None

    @staticmethod
    def _on_cpu(fn):
        return fn()

    @staticmethod
    def _response(row):
        return {"id": row["id"], "intent": "intent", "action": "action",
                "confidence": 0.8, "needs_human": False}

    def handle_many(self, rows, _request_ids):
        return [(self._response(row), {}) for row in rows]

    def handle(self, row, _request_id):
        return self._response(row), {}

    def close(self):
        self.predictor.close()


def _install_fakes(monkeypatch, *, cuda_available=True):
    fake_torch = FakeTorch(cuda_available)
    captured = {}

    class Predictor(FakePredictor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["predictor"] = self

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(V, "_cached_predictions", lambda _tag, _limit: (
        [{"id": "row-1"}],
        {"row-1": {"id": "row-1", "intent": "intent", "action": "action",
                   "confidence": 0.8, "needs_human": False}},
    ))
    monkeypatch.setattr(V, "project", lambda row: row)
    monkeypatch.setattr("experiments.model_exploration.runtime.NeuralEnsemblePredictor", Predictor)
    monkeypatch.setattr("experiments.model_exploration.serve.NeuralTriage", FakeRunner)
    return captured


@pytest.mark.parametrize(("device", "batch_size"), [("cpu", 32), ("cuda:0", 7)])
def test_verify_passes_requested_device_and_batch_size(monkeypatch, device, batch_size):
    captured = _install_fakes(monkeypatch)

    report = V.verify("test-tag", 1, 2, device=device, batch_size=batch_size)

    assert report["pass"] is True
    assert report["requested_device"] == device
    assert report["requested_batch_size"] == batch_size
    assert report["cuda_available"] is True
    assert captured["predictor"].received == ("test-tag", 2, device, batch_size)
    assert report["torch_threads_on_request_worker"] == 2


def test_verify_reports_cuda_unavailable_without_constructing_predictor(monkeypatch):
    captured = _install_fakes(monkeypatch, cuda_available=False)

    report = V.verify("test-tag", 1, 2, device="cuda:0", batch_size=8)

    assert report["pass"] is False
    assert report["cuda_available"] is False
    assert report["requested_device"] == "cuda:0"
    assert report["requested_batch_size"] == 8
    assert report["exception"]["message"] == "CUDA was requested but is not available"
    assert "predictor" not in captured


def test_cpu_report_path_is_unchanged_and_cuda_report_is_separate(tmp_path: Path):
    assert V._report_path(tmp_path, "cpu") == tmp_path / "runtime_parity.json"
    assert V._report_path(tmp_path, "cuda:0") == tmp_path / "runtime_parity_cuda-0.json"
