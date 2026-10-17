"""Standalone inference adapter and local CPU latency probe for a frozen neural ensemble.

This is an experiment module. It does not change the service backend or its defaults.
The torch/transformers imports stay lazy so the standard CPU service can import the module
without installing either package.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.model_exploration import model_common as M  # noqa: E402
from experiments.model_exploration import neural as N  # noqa: E402
from triage import classical as C, text as T  # noqa: E402
from triage.policy import DEFAULT_POLICY, load as load_policy  # noqa: E402


class NeuralEnsemblePredictor:
    """Implements the Predictor methods consumed by Triage for a frozen reference/encoder blend.

    `raw_probs` returns class-aligned action, intent and flow distributions before the service policy prior.
    The neural action head's internal intent feature remains its uncalibrated T=1 softmax; only the exported
    intent head probabilities receive the independently frozen temperature.
    """

    def __init__(self, tag: str, threads: int = 2, device: str = "cpu", batch_size: int = 32):
        import joblib
        import torch

        if threads < 1 or batch_size < 1:
            raise ValueError("threads and batch_size must be positive")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.set_num_threads(threads)

        self.tag = tag
        self.threads = threads
        self.device = device
        self.batch_size = batch_size
        self.output_dir = Path(M.OUT) / tag
        self.model_dir = Path(M.MODEL_DIR)
        self.classical = C.Predictor(self.model_dir / "reference.joblib", dedupe=False)
        checkpoint, tokenizer, metadata, _, _ = N._load_checkpoint(tag, device)
        self.model, self.tokenizer, self.model_metadata = checkpoint, tokenizer, metadata
        self.acts = np.asarray(metadata["labels"]["action"])
        self.icls = np.asarray(metadata["labels"]["intent"])
        self.fcls = np.asarray(metadata["labels"]["flow"])

        ref_classes = (list(map(str, self.classical.acts)), list(map(str, self.classical.icls)),
                       list(map(str, self.classical.fcls)))
        neural_classes = (list(map(str, self.acts)), list(map(str, self.icls)), list(map(str, self.fcls)))
        if ref_classes != neural_classes:
            raise ValueError("reference and neural class orderings differ")

        frozen_path = self.output_dir / "frozen.json"
        if not frozen_path.exists():
            raise FileNotFoundError(f"frozen ensemble settings not found: {frozen_path}")
        self.frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        self.temperatures = {h: float(v) for h, v in self.frozen["temperatures"].items()}
        self.action_classical_weight = float(self.frozen["classical_action_flow_weight"])
        self.intent_classical_weight = float(self.frozen["classical_intent_weight"])

        calibration_path = self.model_dir / f"{tag}-calibration.joblib"
        calibration = joblib.load(calibration_path)
        if "ensemble" not in calibration:
            raise ValueError(f"ensemble calibrator missing from {calibration_path}")
        self.correctness = calibration["ensemble"]["clf"]
        self.thresholds = dict(calibration["ensemble"]["thresholds"])
        self.intent_flow = dict(self.classical.b.get("intent_flow", {}))
        if set(self.icls) - set(self.intent_flow):
            raise ValueError("reference fit-only intent-to-flow map is incomplete")
        # Kept for the existing helper that can resolve a bundle-held map.
        self.b = {"intent_flow": self.intent_flow}
        self.dedupe = False
        self.state_mean = np.asarray(metadata["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(metadata["state_std"], dtype=np.float32)
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        policy = load_policy(DEFAULT_POLICY)
        if policy.version != self.frozen["policy_version"]:
            raise ValueError("loaded policy version differs from frozen model settings")
        self.policy = replace(policy, thresholds=self.thresholds)
        self._prior = self.policy.prior_matrix(self.fcls, self.acts)
        self.metadata = {
            "backend": "neural_ensemble_experiment",
            "tag": tag,
            "model": metadata["model"],
            "revision": metadata["revision"],
            "device": device,
            "threads": threads,
            "batch_size": batch_size,
            "action_classical_weight": self.action_classical_weight,
            "intent_classical_weight": self.intent_classical_weight,
            "temperatures": self.temperatures,
            "calibration": str(calibration_path),
            "frozen_settings": str(frozen_path),
        }

    def state(self, rows) -> np.ndarray:
        """Return the raw compact state features expected by the existing confidence feature model."""
        return C.Predictor.state(rows)

    def _neural_probabilities(self, rows, state) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sequences = N.encode_rows(self.tokenizer, rows, self.model_metadata["max_length"])
        normalized = N._state_norm(state, self.state_mean, self.state_std)
        logits = N._run_logits(self.model, sequences, normalized, len(rows), self.batch_size,
                               self.pad_id, self.device)
        return tuple(C.softmax(logits[head], self.temperatures[head]) for head in ("action", "intent", "flow"))

    def raw_probs(self, rows, st=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = self.state(rows) if st is None else st
        ref_action, ref_intent, ref_flow = self.classical.raw_probs(rows, state)
        neural_action, neural_intent, neural_flow = self._neural_probabilities(rows, state)
        wa = self.action_classical_weight
        wi = self.intent_classical_weight
        return (
            wa * ref_action + (1.0 - wa) * neural_action,
            wi * ref_intent + (1.0 - wi) * neural_intent,
            wa * ref_flow + (1.0 - wa) * neural_flow,
        )

    def confidence(self, pa, pi, pf, st, turn_index) -> np.ndarray:
        features = C._conf_features(pa, pi, pf, st, turn_index)
        return self.correctness.predict_proba(features)[:, 1]

    @staticmethod
    def _customer_probe_row(view: str) -> dict:
        """Undo the exact single-turn marker used by Triage and serialize it like training input."""
        prefix = "zzcust "
        if view.startswith(prefix):
            text = view[len(prefix):]
        else:
            # The public pipeline currently calls with only the customer-only view. Keep a safe fallback for
            # direct callers that pass plain text.
            text = view
        return {"id": "intent-probe", "turn_index": 1,
                "context": [{"speaker": "customer", "text": text}]}

    def intent_probs(self, views: list[str]) -> np.ndarray:
        """Probe the intent head on the latest customer turn alone, preserving the shared Triage signal."""
        w = self.intent_classical_weight
        if w == 1.0:
            return self.classical.intent_probs(views)
        rows = [M.model_input(self._customer_probe_row(v)) for v in views]
        state = self.state(rows)
        _, neural_intent, _ = self._neural_probabilities(rows, state)
        if w == 0.0:
            return neural_intent
        classical_intent = self.classical.intent_probs(views)
        return w * classical_intent + (1.0 - w) * neural_intent

    def close(self) -> None:
        # The caller owns the model lifetime; releasing references permits torch to reclaim its tensors.
        self.model = None


def benchmark(tag: str, threads: int, limit: int, device: str = "cpu") -> dict:
    """Measure single-row predictor work, including the separate last-customer probe (no HTTP server)."""
    if limit < 1:
        raise ValueError("limit must be positive")
    predictor = NeuralEnsemblePredictor(tag, threads=threads, device=device, batch_size=1)
    rows = [M.model_input(r) for r in M.partition("selection")[:limit]]
    if not rows:
        raise ValueError("selection partition is empty")
    policy = predictor.policy
    measurements = []
    try:
        for i in range(5 + len(rows)):
            row = rows[i % len(rows)]
            started = time.perf_counter()
            state = predictor.state([row])
            pa, pi, pf = predictor.raw_probs([row], state)
            after_main = time.perf_counter()
            last_customer = next((t["text"] for t in reversed(row["context"])
                                  if t["speaker"] == "customer" and t["text"].strip()), "")
            view = T.view_all({"context": [{"speaker": "customer", "text": last_customer}]})
            predictor.intent_probs([view])
            after_probe = time.perf_counter()
            pa = C.apply_prior(pa, pf, predictor._prior)
            observed = T.observed_actions(row)
            masked = policy.masked_actions(observed)
            if masked:
                masked_pa = pa[0].copy()
                masked_pa[[j for j, action in enumerate(predictor.acts) if action in masked]] = 0.0
                if masked_pa.sum() > 0:
                    pa[0] = masked_pa / masked_pa.sum()
            predictor.confidence(pa, pi, pf, state, [row["turn_index"]])
            finished = time.perf_counter()
            if i >= 5:
                measurements.append({"main_ms": 1000 * (after_main - started),
                                     "probe_ms": 1000 * (after_probe - after_main),
                                     "including_policy_and_confidence_ms": 1000 * (finished - started)})
    finally:
        predictor.close()

    def quantile(key, q):
        return round(float(np.percentile([r[key] for r in measurements], q)), 2)

    result = {"model": predictor.metadata, "rows": len(measurements), "warmups": 5, "http": False,
              "latency_ms": {key: {"p50": quantile(key, 50), "p95": quantile(key, 95),
                                   "max": round(max(r[key] for r in measurements), 2)}
                             for key in measurements[0]}}
    device_label = device.replace(":", "-")
    out = Path(M.OUT) / tag / f"runtime_bench_{device_label}_threads{threads}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    print(json.dumps(benchmark(args.tag, args.threads, args.limit, args.device), indent=2), flush=True)


if __name__ == "__main__":
    main()
