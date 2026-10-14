# Decisions

Each entry gives the question, the options, the evidence, the choice, and what would change it. Numbers
come from `EXPERIMENT_LOG.md`. Status: **both are provisional.** D1 waits until Phase 1 moves the policy into
a versioned file and Phase 2 sets the routing point. D2 waits for the owner's call on the shipping
question below.

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

**Choice.** Pending the owner's call.
- The rule as written passes B.
- The pre-registration says "adopt only if", which makes the rule a minimum bar, not a guarantee. `analyse.py` reports a pass as "adopt".

The recommendation is **not** to ship B as the default:
- Its whole gain is in one metric (macro-F1) that the CPU model already clears with a wide margin (0.70 against a bar of 0.50).
- It adds 3.3 s p95 and an outside dependency to every request.
- It costs $0.31–1.08 per 1,000 messages.

The two models' different strengths point to a better use: combine them per flow or by probability. That is a new hypothesis and needs a fresh test.

**What would change it.** Either of these would justify a scoped LLM step:
- A confirmation test on held-out train rows (fold B, never seen by any model choice) that reproduces Luna-on-the-least-confident-20% at +2 points of action.
- A Haiku–CPU combination that lifts intent accuracy, not just macro-F1.
