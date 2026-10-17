"""No-Torch tests for the optional neural experiment's shared HTTP pipeline adapter."""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from experiments.model_exploration import serve
from triage import prompts as P
from triage import service
from triage import text as T
from triage.policy import DEFAULT_POLICY, load


class FakePredictor:
    def __init__(self, model_dir: Path):
        policy = load(DEFAULT_POLICY)
        self.tag = "fake-neural"
        self.model_dir = model_dir
        self.thresholds = {"low": 0.95, "high": 0.95}
        self.frozen = {"policy_version": serve.FROZEN_POLICY_VERSION}
        self.metadata = {"backend": "neural_ensemble_experiment", "tag": self.tag,
                         "model": "unit-test/tiny-bert", "revision": "local",
                         "device": "cpu", "threads": 1, "batch_size": 1,
                         "action_classical_weight": 0.5, "intent_classical_weight": 0.5}
        self.acts = np.asarray(T.ACTIONS30)
        self.icls = np.asarray(P.INTENTS)
        self.fcls = np.asarray(sorted(policy.flow_permitted))
        self.intent_flow = {intent: "account_access" for intent in self.icls}
        self.action = "verify-identity"
        self.calls = 0
        self.closed = False

    def state(self, rows):
        return np.zeros((len(rows), len(T.STATE_KEYS)), dtype=np.float32)

    def raw_probs(self, rows, st=None):
        self.calls += len(rows)
        pa = np.zeros((len(rows), len(self.acts)), dtype=np.float32)
        pa[:, list(self.acts).index(self.action)] = 1.0
        pi = np.zeros((len(rows), len(self.icls)), dtype=np.float32)
        pi[:, 0] = 1.0
        pf = np.zeros((len(rows), len(self.fcls)), dtype=np.float32)
        pf[:, list(self.fcls).index("account_access")] = 1.0
        return pa, pi, pf

    def confidence(self, pa, pi, pf, st, turn_index):
        return np.full(len(turn_index), 0.9, dtype=np.float32)

    def intent_probs(self, views):
        result = np.zeros((len(views), len(self.icls)), dtype=np.float32)
        result[:, 0] = 1.0
        return result

    def close(self):
        self.closed = True


def _fake(model_dir: Path) -> FakePredictor:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "reference.joblib").write_bytes(b"test reference artifact")
    (model_dir / "fake-neural").mkdir(exist_ok=True)
    (model_dir / "fake-neural" / "best.pt").write_bytes(b"test neural checkpoint")
    return FakePredictor(model_dir)


def _request():
    return {"id": "neural-service-row", "turn_index": 1,
            "context": [{"speaker": "customer", "text": "I need help with my account please."}]}


def test_frozen_policy_version_and_calibrated_thresholds_drive_triage(monkeypatch, tmp_path):
    predictor = _fake(tmp_path / "models")
    original_policy = load(DEFAULT_POLICY)
    monkeypatch.setattr(serve, "load_policy", lambda path: replace(original_policy, version="wrong-version"))
    with pytest.raises(ValueError, match="requires frozen policy"):
        serve.NeuralTriage(predictor)

    monkeypatch.setattr(serve, "load_policy", load)
    triage = serve.NeuralTriage(predictor)
    try:
        frozen = triage.store.current()
        assert frozen is triage.store.current()
        assert frozen.version == serve.FROZEN_POLICY_VERSION
        assert frozen.thresholds == predictor.thresholds
        assert triage.settings.route_second_intent is False
        assert triage.llm is None

        response, trace = triage.handle(_request(), "test-request")
        assert response["action"] == predictor.action
        assert response["needs_human"] is True
        assert "low_confidence" in trace["route_reasons"]
        assert trace["policy_version"] == serve.FROZEN_POLICY_VERSION
    finally:
        triage.close()
    assert predictor.closed


def test_health_reports_cached_neural_artifact_provenance_without_classical_bundle_claim(monkeypatch, tmp_path):
    predictor = _fake(tmp_path / "models")
    hashed = []
    real_hash = serve._file_sha256

    def record_hash(path):
        hashed.append(Path(path))
        return real_hash(path)

    monkeypatch.setattr(serve, "_file_sha256", record_hash)
    triage = serve.NeuralTriage(predictor)
    try:
        health = triage.health()
        second_health = triage.health()
        assert health["backend"] == "neural_ensemble_experiment"
        assert second_health["model"]["checkpoint_sha256"] == health["model"]["checkpoint_sha256"]
        assert len(hashed) == 2
        assert set(hashed) == {tmp_path / "models" / "reference.joblib",
                               tmp_path / "models" / "fake-neural" / "best.pt"}
        assert "bundle_sha256" not in health
        assert health["model"]["model"] == "unit-test/tiny-bert"
        assert health["model"]["revision"] == "local"
        assert health["model"]["checkpoint_sha256"]
        assert health["model"]["reference_bundle_sha256"]
        assert health["policy"]["thresholds"] == predictor.thresholds
        assert health["policy"]["threshold_source"] == "frozen neural ensemble calibration"
        assert health["llm"] == {"enabled": False, "mode": "off"}
        assert health["routing"]["route_second_intent"] is False
    finally:
        triage.close()


def test_create_app_lifespan_loads_and_serves_the_neural_pipeline(tmp_path):
    predictor = _fake(tmp_path / "models")
    loaded = []

    def loader():
        triage = serve.NeuralTriage(predictor)
        loaded.append(triage)
        return triage

    with TestClient(service.create_app(loader=loader, exit_on_load_error=True)) as client:
        deadline = time.monotonic() + 3
        while (health := client.get("/healthz")).status_code != 200 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert health.status_code == 200
        assert health.json()["backend"] == "neural_ensemble_experiment"
        response = client.post("/triage", json=_request())
        assert response.status_code == 200
        assert response.json()["id"] == "neural-service-row"
        assert response.json()["action"] == predictor.action
        assert predictor.calls == 1

    assert len(loaded) == 1
    assert predictor.closed
