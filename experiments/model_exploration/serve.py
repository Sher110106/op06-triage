"""Optional HTTP service for the frozen neural ensemble experiment.

Run with one uvicorn worker after producing a checkpoint and frozen calibration artifacts:

    TRIAGE_NEURAL_TAG=modernbert_joint TRIAGE_NEURAL_THREADS=2 \
      TRIAGE_NEURAL_DEVICE=cpu uvicorn experiments.model_exploration.serve:app \
      --host 127.0.0.1 --port 8000 --workers 1

This adapter reuses the production request pipeline but keeps the experiment opt-in. It
freezes the calibrated cuts and policy version at process load, and it does not enable the
LLM action step or second-intent routing used in the controlled model comparison.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import math
import os
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from triage import classical as C
from triage.llm_action import LLMConfig
from triage.pipeline import Settings, Triage
from triage.policy import DEFAULT_POLICY, Policy, load as load_policy
from triage.service import create_app

FROZEN_POLICY_VERSION = "2026-10-17.2"


class FrozenPolicyStore:
    """PolicyStore-compatible immutable snapshot; current() always returns the same object."""

    def __init__(self, policy: Policy, path: str | Path):
        self._policy = policy
        self.path = Path(path)
        self.last_error: str | None = None
        self.loaded_at = time.time()

    @property
    def version(self) -> str:
        return self._policy.version

    def current(self) -> Policy:
        return self._policy


def _file_sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


class NeuralTriage(Triage):
    """Existing Triage stages wired to a frozen NeuralEnsemblePredictor instance."""

    def __init__(self, predictor, *, policy_path: str | Path = DEFAULT_POLICY):
        policy = load_policy(policy_path)
        frozen_version = getattr(predictor, "frozen", {}).get("policy_version")
        if policy.version != FROZEN_POLICY_VERSION or frozen_version != FROZEN_POLICY_VERSION:
            raise ValueError(
                f"neural experiment requires frozen policy {FROZEN_POLICY_VERSION}; "
                f"loaded policy={policy.version!r}, model policy={frozen_version!r}")

        thresholds = getattr(predictor, "thresholds", None)
        if (not isinstance(thresholds, dict) or set(thresholds) != set(policy.thresholds)
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) or not 0 <= v <= 1 for v in thresholds.values())):
            raise ValueError("neural calibration must provide finite [0, 1] thresholds for every policy tier")
        effective_policy = replace(policy, thresholds=dict(thresholds))

        # Initialize only the shared Triage orchestration state. The predictor already owns
        # its reference model; loading another production Predictor would waste memory.
        self.started_at = time.time()
        self.cpu = predictor
        self.store = FrozenPolicyStore(effective_policy, policy_path)
        self.settings = Settings(route_second_intent=False)
        self.intent_flow = dict(predictor.intent_flow)
        self._aidx = {str(action): index for index, action in enumerate(predictor.acts)}
        self._prior_cache = None
        self.llm_cfg = LLMConfig(mode="off")
        self.llm = None
        self.llm_error = None
        self._pool: cf.ThreadPoolExecutor | None = None
        self._pool_lock = threading.Lock()
        self._cpu_thread: threading.Thread | None = None
        self._cpu_exec = cf.ThreadPoolExecutor(1, thread_name_prefix="triage-neural-cpu",
                                               initializer=self._cpu_init)
        self.predictor = predictor

        model_dir = Path(getattr(predictor, "model_dir", C.DEFAULT_BUNDLE.parent))
        self._reference_path = model_dir / "reference.joblib"
        self._checkpoint_path = model_dir / str(getattr(predictor, "tag", "")) / "best.pt"
        # Artifacts are immutable for this process. Hash them once after the model loads,
        # rather than rereading hundreds of megabytes on every health probe.
        self._reference_bundle_sha256 = _file_sha256(self._reference_path)
        self._checkpoint_sha256 = _file_sha256(self._checkpoint_path)
        self._metadata = dict(getattr(predictor, "metadata", {}))

    def health(self) -> dict:
        policy = self.store.current()
        model = dict(self._metadata)
        model.update({"backend": "neural_ensemble_experiment",
                      "checkpoint": str(self._checkpoint_path),
                      "checkpoint_sha256": self._checkpoint_sha256,
                      "reference_bundle": str(self._reference_path),
                      "reference_bundle_sha256": self._reference_bundle_sha256})
        return {
            "status": "ok",
            "backend": "neural_ensemble_experiment",
            "model": model,
            "policy": {"version": policy.version, "path": str(self.store.path),
                       "thresholds": dict(policy.thresholds),
                       "threshold_source": "frozen neural ensemble calibration",
                       "last_error": self.store.last_error,
                       "loaded_at": datetime.fromtimestamp(self.store.loaded_at, timezone.utc).isoformat()},
            "llm": {"enabled": False, "mode": "off"},
            "routing": {"route_second_intent": False, "route_off_policy": self.settings.route_off_policy},
            "started_at": datetime.fromtimestamp(self.started_at, timezone.utc).isoformat(),
        }

    def close(self) -> None:
        super().close()
        close = getattr(self.predictor, "close", None)
        if close is not None:
            close()


def _load_neural_from_env():
    tag = (os.environ.get("TRIAGE_NEURAL_TAG") or "").strip()
    if not tag:
        raise ValueError("TRIAGE_NEURAL_TAG is required for the neural experiment service")
    threads = int(os.environ.get("TRIAGE_NEURAL_THREADS", "2"))
    device = (os.environ.get("TRIAGE_NEURAL_DEVICE") or "cpu").strip()
    batch_size = int(os.environ.get("TRIAGE_NEURAL_BATCH_SIZE", "16"))
    from experiments.model_exploration.runtime import NeuralEnsemblePredictor

    predictor = NeuralEnsemblePredictor(tag, threads=threads, device=device, batch_size=batch_size)
    try:
        return NeuralTriage(predictor)
    except BaseException:
        predictor.close()
        raise


app = create_app(loader=_load_neural_from_env, exit_on_load_error=True)
