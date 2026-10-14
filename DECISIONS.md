# Decisions

Each entry gives the question, the options, the evidence, the choice, and what would change it. Numbers
come from `EXPERIMENT_LOG.md`. Status: **D2 is decided** by a pre-registered confirmation test. **D1 is provisional** until
Phase 1 moves the policy into a versioned file and Phase 2 sets the routing point.

## D1. The 9% of gold actions that break the written procedure

**Question.** About 9% of gold next actions fall outside what the flow's written procedure permits
(8.8% literal, 5.7% once the `membership-privileges` / `membership` naming mismatch is resolved). Should
the service predict what agents do, enforce what the procedure says, or something in between?

**Options.**
1. **Follow the label.** Learn from gold actions only, with no policy input. This gives the best raw accuracy, but the service will recommend off-policy actions with no warning.
2. **Follow the policy, hard.** Mask out every action the predicted flow does not permit. Violations are near zero, but when the flow prediction is wrong (12% of rows), the right action gets masked out.
3. **Soft policy prior, and flag the conflict.** Multiply action probabilities by the expected permission under the flow probabilities. Keep the alias. Lower the confidence, and route to a person when the top action is off-policy.

**Evidence (R4).**
- The soft prior with the alias gave the best action accuracy of the masks tried.
- It brought the policy-violation rate to 0.080 literal and 0.045 with the alias. Both are below the gold's own rates (0.088 and 0.057).
- 50 baseline "violations" came only from the naming mismatch, and the alias removes all but 5 of them.

**Choice (provisional).** Option 3.
- Phase 1 moves the flow → permitted-actions table and the alias into a versioned policy file, so a procedure change becomes a file change with a replay over dev rather than a retrain.
- Phase 2 adds the "off-policy → route to a person" rule, and measures what it costs in routed share.

**What would change it.** If a procedure change made the soft prior keep recommending a newly forbidden
action (the interview's procedure-change drill), the remedy is a hard mask for that action alone, applied
from the policy file.

## D2. Does an LLM belong in the v1 service?

**Question.** The CPU model already passes every bar. Is an LLM worth its cost, its latency and the new
ways it can fail? The user assumes the reviewer's gateway routes all providers, so the option is open.

**Options** (pre-registered in `experiments/llm_tail/PREREGISTRATION.md`, pushed before any call):
- **None.** The CPU model only.
- **A.** GPT-6 Luna answers the rows the CPU model would route to a person (34.7%).
- **B.** Claude Haiku 5.5 picks the intent on every row. The CPU model keeps the action and confidence.
- **C.** A and B together.

**Evidence** (`experiments/llm_tail/results.json`):
- The primary analysis uses the 1,940 dev rows the earlier probe never saw.
- Intervals are 95%, from a paired conversation-level bootstrap.
- All 4,600 calls succeeded. One Luna output was invalid JSON, and the fallback handled it.

| System | Intent acc | Intent macro-F1 | Action acc | Rule |
|---|---|---|---|---|
| A. Luna on routed rows | +0.000 [−0.013, +0.012] | +0.017 [−0.003, +0.037] | +0.014 [−0.001, +0.029] | fails (action gain < 0.02, interval includes 0) |
| B. Haiku intent | −0.003 [−0.024, +0.018] | **+0.031 [+0.004, +0.054]** | 0 | **passes on macro-F1** |
| C. Both | −0.003 | +0.031 | +0.014 | fails (A's part fails) |

| | p50 / p95 per call | Cost per 1,000 messages (list, with caching / uncached) |
|---|---|---|
| CPU model | 8 / 15 ms | $0 |
| Luna, every row | 1.5 / 2.8 s | $0.21 / $0.63 |
| Haiku, every row | 1.7 / 3.3 s | $0.31 / $1.08 |

What B's gain is made of:
- Intent accuracy doesn't move. Macro-F1 rises because Haiku does better on rare intents: mean per-class F1 +0.047 over the 40 intents with fewer than 15 rows, against +0.023 over the rest.
- The two models are strong in different flows:
  - Haiku is better on manage_account (+11.5 points), purchase_dispute (+8.1) and product_defect (+6.2).
  - It is worse on shipping_issue (−11.4) and subscription_inquiry (−10.9).
- On noisy rows, Haiku's intent is 1.3 points worse.
- Action, which the manager acts on, can't change under B by construction.

An unplanned finding:
- Sending a smaller share (the least-confident 20%) to Luna gave action +0.024 [+0.012, +0.035] on the same rows.
- That share was picked after seeing the results, so it is a hypothesis, not evidence. It needs a fresh test.

**First result.** The dev test passed B on macro-F1 alone, and flagged two leads. Shipping B wasn't
recommended: intent accuracy stayed flat, and it added an LLM call to every row. Both leads were found by
looking at dev, so they needed fresh rows.

**Confirmation test** (`experiments/llm_confirm/`, pre-registered and pushed before any call):
- **Rows:** 3,966 rows of train fold B, which no model choice had seen.
- **Leakage measures:** the CPU confidences are out-of-fold, and the LLM examples were retrieved from fold A only.
- **Frozen rules, all fixed from dev:**
  - Luna's action when the CPU confidence is below 0.642841 (16.7% of fold-B rows).
  - The LLM's intent when the CPU model's p(intent) is below 0.95 and the LLM's intent is among the CPU model's top 3.

| Step | Intent acc | Action acc | Top-70 ratio | Rule (+0.020, interval above 0) |
|---|---|---|---|---|
| Luna action on low-confidence rows | 0 | **+0.025 [+0.017, +0.032]** (0.833 → 0.858) | 0.324 → 0.380 | **passes** |
| Luna intent, gated, top 3 | +0.014 [+0.004, +0.024] | 0 | 0 | fails (too small) |
| Haiku intent, gated, top 3 | +0.013 [+0.003, +0.023] | 0 | 0 | fails (too small) |

**Choice.** v1 ships the **Luna action step**, behind a config flag and with the CPU answer as the
fallback. Nothing else in v1 uses an LLM.

| | Value |
|---|---|
| Rows that call the LLM | 17–20%. The rest are answered by the CPU model in about 15 ms. |
| Cost per 1,000 messages at list price | $0.03 with caching, $0.11–0.13 uncached |
| Latency (offline estimate from single calls) | p50 15 ms, **p95 about 1.8 s**, p99 about 2.7 s. The p95 lands inside the LLM rows, because more than 5% of rows call it. |
| Confidence ordering | Slightly worse (top-70 ratio 0.32 → 0.38), still far under the 0.85 bar |
| Pinning | `openai/gpt-6-luna-20260922`, OpenAI host only, fallbacks off, reasoning off |

**What the intent step shows.** Its gain is real (the interval is above zero) but under the bar set in
advance, so it stays out. Writing this down instead of lowering the bar after the fact is the point of
pre-registering.

**What would change it.**
- If the latency budget can't absorb a p95 near 2 s, run the step on fewer rows (dev showed +0.011 at 10%) or turn the flag off. The service stays correct either way.
- If the provider is down or slow, the timeout returns the CPU answer, and that degradation should show up in monitoring, not in errors.
- If Phase 3's load test finds tail latency much worse than these single-call numbers, revisit the share.
