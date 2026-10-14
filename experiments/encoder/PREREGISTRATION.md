# Pre-registration: is a fine-tuned small encoder a better choice than the LLM step?

Written on 14 Oct 2026 while the encoder was still training, before any fold-B or dev output existed.

## Why

The 13 Oct probe fine-tuned `jhu-clsp/ettin-encoder-32m` (32M parameters) on a CPU. It scored dev action
0.834 against the classical model's 0.810, but on the probe's own scorer, trained on 90% of all train. That
leaves two questions, which this test answers on rows no model saw:
1. Is the encoder really better than the classical CPU model?
2. Could a CPU-only ensemble do as well as the LLM action step (`DECISIONS.md` D2)? If so, every external call goes away.

## Setup

- **Training:** fold A only, the same seed-0 split as `triage.classical.folds()` (`ft_foldA.py`). Early stopping uses 10% of fold-A conversations. Hyperparameters are those of the probe run: 2 epochs, lr 1e-4, batch 32, max 256 tokens, 16 CPU threads on the remote box.
- **Outputs:** temperature-scaled intent, flow and action probabilities for fold B (3,966 rows), dev, dev_noisy and hostile.
- **Ensemble weights:** chosen on dev (2,300 rows) only, from the grid 0.0, 0.1, …, 1.0.
  - Action: p = w · p_classical + (1 − w) · p_encoder, where p_classical is the classical model's action probability after the policy prior, as served. w maximises dev action accuracy.
  - Intent: the same form, with its own weight v, maximising dev intent accuracy.

## Systems compared on fold B

All are paired, with a 95% conversation bootstrap (`harness.eval.paired`).

| Name | System |
|---|---|
| C0 | Classical CPU model (`experiments/llm_confirm/cpu_B_predictions.jsonl`) |
| S0 | The shipped v1 system: C0 plus the Luna action step on CPU confidence < 0.642841 (`L_act` in `../llm_confirm/results_B.json`) |
| E1 | The encoder alone (argmax of its heads) |
| E2 | The CPU ensemble with the dev-chosen weights |

## Decision rule

- **The encoder is a real CPU gain** if E2's action accuracy beats C0's by at least 0.020, with the interval's lower bound above 0.
- **E2 replaces the LLM step in v1** only if its action accuracy beats S0's by at least 0.020, with the interval's lower bound above 0. Even then, the switch first needs a confidence model and routing thresholds refit for the ensemble, checked on dev, before it ships.
- **If E2 is within 0.5 points of S0** (lower bound of E2 − S0 above −0.005) but not 2 points better, it becomes the recommended v2 path: same accuracy, no external calls. v1 stays as decided in D2.
- **Otherwise,** the encoder is logged as a measured alternative and rejected for v1. The cost of adopting it is a PyTorch runtime (about 700 MB installed), a 122 MB model and about 80 ms p95 per request on one CPU thread.
