# Pre-registration: a confirmation test on train fold B

Written and pushed on 14 Oct 2026 **before any API call of this test**.

## Why this test

The full-dev test (`../llm_tail/`) passed no action step. It also left two leads, both found by looking
at dev results, so neither is evidence yet:

1. **Luna's action on the least-confident 20% of rows** gave +0.024 [+0.012, +0.035] on dev.
2. **The CPU model and the LLMs are strong in different flows.** On dev, with 5-fold cross-validation by conversation (`rule_selection_dev.json`), the best combination was: when the CPU model's top intent probability is below 0.95, take the LLM's intent if it is among the CPU model's top 3. Out-of-fold intent accuracy: 0.709 → 0.750 with Haiku, 0.747 with Luna.

This test checks both on rows that played no part in choosing them.

## Rows

- **Train fold B:** 3,966 rows, the 20% of train conversations held out from the CPU model's intent, flow and action fitting (seed-0 split, `triage.classical.folds()`).
- **What fold B was used for:** fitting the confidence stacker, the temperatures and the routing threshold. It played no part in choosing the LLM prompt or either rule tested here.

Measures against leakage:
- CPU confidence on fold B is **out-of-fold**: a 5-fold cross-fit by conversation, so the stacker never scores rows it was fit on. `prep.py` asserts that this reproduces training's threshold exactly.
- The LLMs' retrieved examples come from **fold A only**, so a fold-B row cannot retrieve itself or its own conversation. This is a slightly smaller index than the dev test used.

Limits:
- Fold B is clean text only, with no noisy twins. The noisy-gap bar can't be checked here.

## Models

Same as the dev test, with the same V3 prompt:

| Model | Snapshot | Settings | Rows |
|---|---|---|---|
| GPT-6 Luna | `openai/gpt-6-luna-20260922` | OpenAI host pinned, fallbacks off, reasoning none | every fold-B row |
| Claude Haiku 5.5 | `claude-haiku-5-5` | effort low | only the 2,060 rows the frozen intent rule reads (CPU p(intent) < 0.95) |

## Frozen rules

These constants were all fixed before this test, from dev.

| Step | Rule |
|---|---|
| **Action step** (`L_act`) | If the CPU confidence is below **0.642841** (dev's 20% quantile), take Luna's action. Confidence becomes (CPU confidence + agreement) / 2. Intent stays the CPU model's. |
| **Intent step** (`L_int` with Luna, `H_int` with Haiku) | If the CPU model's top intent probability is below **0.95**, take the LLM's intent, but only if it is among the CPU model's **top 3**. |
| **Fallback** | Failed, invalid or `unknown`/`none` LLM answers leave the CPU answer in place. |
| **Joint system** (`L_both`) | `L_act` plus `L_int`: one Luna call, made only when either gate opens. |

## Decision rule

Comparisons are paired against the CPU model on all 3,966 rows, with a 95% conversation-level bootstrap (2,000 draws, seed 7).

- **Adopt the action step** only if action accuracy rises by at least 0.020 and the interval's lower bound is above 0.
- **Adopt the intent step** only if intent accuracy rises by at least 0.020 and the interval's lower bound is above 0.
  - Luna is checked first. If Luna passes, use Luna: one provider, one call per row, cheaper and faster.
  - Use Haiku only if Haiku passes and Luna does not.
- **If a step passes,** v1 ships it behind a config flag, with a hard timeout falling back to the CPU answer. Its cost and latency must then be measured under load in Phase 3.
- **If neither passes,** v1 is CPU-only and D2 closes as "no LLM".
