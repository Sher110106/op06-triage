# Pre-registration: does an LLM belong in the v1 service?

Written and committed on 14 Oct 2026 **before any API call of this test**. The commit that adds this
file comes before the commit that adds the run outputs (`runs/`, `ledger.jsonl`).

## Why this test

On the 13 Oct probe (360 labelled rows), no LLM beat the CPU model on next action, and the intent gains had
intervals that included zero. Since the user now assumes the reviewer's gateway routes all providers, two
LLM roles are possible. This test settles them on the full development set rather than a sample.

## Systems

All share the CPU model `dev-classical-v0` (`results/raw/dev-classical-v0/predictions.jsonl`, trained on
train only). Both LLMs get the probe's V3 prompt unchanged: label descriptions, procedures, the 8 most
similar train conversations, and the parsed last-action hint. The static prompt's SHA-256 starts with
`3511a0a0072bdcbf`, identical to the probe's.

| System | What changes from the CPU model |
|---|---|
| **A. Luna on the uncertain rows** | Rows the CPU model would route to a human (confidence below 0.8498, the 30% quantile of train hold-out confidences, fixed before dev was seen) take GPT-6 Luna's intent and action. |
| **B. Haiku for intent** | Every row takes Claude Haiku 5.5's intent. Action, confidence and needs_human stay the CPU model's. |
| **C. Both** | B's intents everywhere, A's actions on the uncertain rows. Scored from the same calls; no new calls. |

Models and settings:
- GPT-6 Luna: `openai/gpt-6-luna-20260922` through OpenRouter, provider pinned to OpenAI with fallbacks off, reasoning effort `none`.
- Claude Haiku 5.5: `claude-haiku-5-5` on the Anthropic API, effort `low`, prompt caching on the static prompt.
- Luna runs on all 2,300 rows so other escalation shares can be studied, but the decision uses only the rule above.

Rules shared by every system:
- **Fallback:** if a call fails, fails to parse, falls outside the label lists, or returns `unknown`/`none` on a labelled row, the CPU answer stands.
- **Confidence after escalation (A, C):** on escalated rows, confidence = (CPU confidence + agreement) / 2, where agreement is 1 when the LLM's action equals the CPU action. Other rows keep the CPU confidence. This formula was the best on the probe and is fixed here.
- **needs_human:** confidence below 0.8498, as in the CPU model.
- Gold labels are used only to score.

## Rows

- **Primary: 1,940 fresh rows.** These are the 2,300 labelled dev and dev_noisy rows minus the 360 the probe already used to form the hypothesis (`seen_in_probe_ids.json`).
- **Secondary: all 2,300 rows.** These are reported, but they don't decide.

## Decision rule

Comparisons are paired against the CPU model, with 95% intervals from a conversation-level bootstrap (2,000 draws, seed 7, `harness.eval.paired`).

- **Adopt A** only if action accuracy on the fresh rows rises by at least 0.020 and the interval's lower bound is above 0.
- **Adopt B** only if intent accuracy or intent macro-F1 on the fresh rows rises by at least 0.020, with the interval's lower bound above 0.
- **Adopt C** only if A's rule and B's rule both pass when applied to C.
- **Applies to every option:** an adopted system must still pass every grader bar on all 2,300 rows.
- **If nothing passes,** v1 ships with no LLM, and this measurement is the evidence.

These are also reported, but they don't decide: cost per 1,000 messages at list price (uncached and with caching), p50 and p95 latency of the LLM calls, results per noise kind, the top-70% error ratio, other escalation shares (10–50% and 100%), and Luna-for-intent.

## What adoption would also require (Phase 1, not this test)

- A hard per-call timeout, falling back to the CPU answer.
- Enum checks on every output.
- Pinned snapshots and hosts.
- Cost and latency measured under load.
- Recorded failure behaviour when the provider is down.
