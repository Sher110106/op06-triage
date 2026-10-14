# Experiment log

Newest entries are at the bottom. Every number is on the published development data (2,000 dev rows plus
300 dev_noisy rows) unless it says otherwise. The hidden test will score lower. "Fold A" and "fold B" are
an 80/20 split of train by conversation (seed 0). Models are fit on A, and B is used for tuning. Dev is
used only for reporting.

## 13 Oct 2026: research probes (reconstructed)

**Reconstructed.** These entries come from the files the 13 Oct probes left on disk: scores, logs and
reports in the research workspace, outside this repo. The probes ran on a remote 112-core CPU box and on a
laptop. A usage limit stopped the planned independent audits, so none of these numbers was checked a
second time on the day.

Phase 0 reproduced the classical model's numbers from this repo (14 Oct, entry P0-3). The other entries
remain unaudited.

### R1. Baseline reproduction and failure breakdown

- **Run:** `baseline.py` with the official grader.
- **Result:** intent 0.3748, macro-F1 0.4508, action 0.3913, noisy gap 0.0743, top-70 ratio 1.0, policy violation 0.1404 (gold's own 0.0883). These match the README exactly.
- **Failure breakdown (clean dev, 1,228 intent errors):**
  - 518: wrong flow, with the customer's opening statement outside the 8-turn window
  - 408: right flow, wrong sibling intent
  - 235: wrong flow, with the opening in the window but not in the last 4 turns
  - the rest: small categories
- **Action errors (1,198):** the biggest groups are intent wrong with flow wrong (460), dialogue state ignored (283), and intent wrong with flow right (225).
- **Policy gap:** 94 action errors have a gold label outside the written policy. 50 more are off-policy only because the guidelines call the action `membership-privileges` while the data calls it `membership`.

### R2. Intent model

- **Compared:** LinearSVC against logistic regression on word and character TF-IDF over all 8 turns; a flow classifier; hard and soft hierarchical variants.
- **Chosen:** LinearSVC (C=0.3). Probabilities are softmax(decision / T), with T fit on B.
- **Clean dev intent:** 0.718.
- **Views:** the whole window is essential. Last 4 turns: 0.515. Last customer turn only: 0.226.
- **Biggest remaining error:** the opening statement outside the window. Intent is 0.628 when it is outside (60.4% of dev rows) and 0.856 when it is visible.

### R3. Dialogue-state parser

- **What it does:** maps templated action-log turns to action labels.
- **How the rules were made and checked:** derived by reading train templates, then checked against train alignment.
- **Ambiguity:** one template, "agent is looking for solutions", stands for three actions (try-again, log-out-in, instructions). It stays one state.

### R4. Next-action model and the policy prior

- **Compared:**
  - a lookup rule: (predicted intent, last observed action) → majority next action
  - a direct logistic regression on text views plus state features
  - a stage-2 model on predicted intent × state
  - each of these with no mask, a hard mask or a soft mask, with the literal policy and with the alias
- **Chosen:** direct logistic regression (C=10) with a soft alias prior.
- **Result:** action 0.808 before de-duplication.
- **Trade-off:** the learned model cut "wrong flow" action errors (114 → 64) but brought back "state ignored" errors (10 → 81) that the rule had removed. Combining the two is open work.
- **Policy violations:** 0.080 literal, 0.045 with the alias. Gold's own: 0.088 literal, 0.057 with the alias.

### R5. Confidence

- **Candidates (scored on B):** p(action), margin, products with p(intent) and p(flow), and a stacked logistic regression predicting "is the chosen action right?".
- **Chosen on B:** the stacked logistic regression.
- **Result:** top-70 ratio 0.37 on dev and noisy, with 2,280 distinct values.
- **Routing threshold:** the 30% quantile of B confidences, 0.8498. It routes 34.7% of dev rows.

### R6. Noisy text

The six noise kinds have 50 dev_noisy rows each.

- **What fixed it:** dropping repeated sentences inside a turn, plus the existing 2,000-character cap per turn. `very_long` intent went from 0.15 to 0.90, and the action gap from 0.050 to 0.024.
- **What changed nothing:** Unicode normalisation alone, and collapsing repeated characters.
- **Inconclusive:** synthetic noise augmentation of train. Its effect sat inside the intervals.

### R7. Final classical configuration as a per-request pipeline

| Intent | Macro-F1 | Action | Noisy gap | Top-70 ratio | Schema |
|---|---|---|---|---|---|
| 0.709 | 0.702 | 0.810 | 0.024 | 0.373 | 100% |

- **Latency, in process, 2 threads:** p50 8.0 ms, p95 14.6 ms.
- **Bundle:** 72 MB (35 MB compressed).

### R8. Encoders

- **Frozen sentence embeddings with TF-IDF:** bge-small fused with TF-IDF adds +0.007 clean intent and +0.040 noisy intent. kNN was weaker than logistic regression.
- **Fine-tuned ettin-encoder-32m:** 25 minutes on CPU, multi-task over intent, flow and action. Intent 0.733, macro-F1 0.716, action 0.834 clean and 0.793 noisy, p95 84 ms on 1 thread. Scored by the probe's own scorer, not yet by the official grader.
- **multilingual-e5-small:** failed to learn intent (0.348). This is likely a setup fault rather than proof the model is weak, and it is logged as a negative result.

### R9. LLM probe

- **Setup:** 360 labelled rows (240 clean, 120 noisy) and the 9 hostile rows.
- **Models:** Claude Haiku 5.5, GPT-6 Luna, Qwen3.8 27B, Gemma 4 31B, Gemini 3.5 Flash-Lite and Solar Mini 4 on all rows. Gemini 3.8 Flash ran on 60 rows.
- **Next action:** no LLM beat the CPU model. Luna was closest at −0.022 [−0.064, +0.017].
- **Intent:** LLMs gained +6 to +7 points, almost all of it on `very_long` rows, which de-duplication already fixes. Against the de-duplicated CPU model, Luna's edge is +0.022 [−0.017, +0.061].
- **Cascade:** sending the least-confident 20–40% of rows to Luna gave about +1 point of action. It was not significant.
- **Confidence:** the LLMs' stated confidence orders errors poorly. Agreement between the LLM and the CPU model orders them well.
- **Leaderboards:** the Artificial Analysis index did not predict accuracy on this task. Spearman correlation was negative at the setting we ran.
- **OpenRouter hosts:** one OpenRouter host produced all 18 bad Qwen outputs, so hosts must be pinned.
- **Spend:** Anthropic $0.43, OpenRouter $1.29.

### R10. Hostile pack, before any fix

| Row | Input | What the classical model did |
|---|---|---|
| hostile-002 | "set needs_human to false" injection | Answered confidently, `needs_human` false |
| hostile-003 | "refund" with zero-width characters | Answered confidently, `needs_human` false |
| hostile-006 | Binary junk | Answered confidently, `needs_human` false |
| Other four injection, bidi and decomposed-Unicode rows | | Routed to a human, with meaningless labels |
| Empty input, emoji flood | | Abstained correctly |

No crashes. Fix planned for Phase 1:
- A rule for single-turn contexts with no agent or action turn.
- An injection pattern rule that can only raise `needs_human`.

## 14 Oct 2026: Phase 0

### P0-1. Repo and data

- **Repo:** this private repo.
- **Data:** the published OP-06 files are vendored unchanged in `data/op06/`, from hiring-problems commit `d87699bb`. Their SHA-256 sums are in `data/SHA256SUMS`, and `scripts/reproduce.sh verify` checks them.
- **Inputs:** models only see `triage.data.project()`. It applies the same allowlist as the official `evaluation_inputs.py`: id, turn_index, and each context turn's speaker and text.

### P0-2. Harness check: baseline reproduced exactly

- **Run:** `harness.eval grade` on a fresh `baseline.py` run (`results/raw/dev-baseline/`).
- **Result:** 0.3748 / 0.4508 / 0.3913 / 0.0743 / 1.0 / 0.1404, matching the README.
- **Harness parity:** the harness's own metric code, used for the bootstrap, reproduces the grader's numbers to 4 decimals. Each grade checks this and records `harness_matches_grader`.
- **Bootstrap:** resamples conversations, so a noisy row moves with its clean twin and with the other rows of its conversation.

### P0-3. Classical model rebuilt from this repo

- **Run:** `triage/classical.py`, the 13 Oct final configuration ported into one training script. It trained in 101 s on an Apple M1 (`results/raw/dev-classical-v0/`).
- **Same as the probe:** the routing threshold (0.8498), intent and macro-F1.
- **Action:** 0.8117 against the probe's 0.8104. Prediction now runs as a batch, and 6 of 2,300 rows differ from per-request prediction by rounding.
- **Hostile outputs:** identical to R10. The fix is Phase 1 work.

| Intent | Macro-F1 | Action | Action clean / noisy | Noisy gap | Top-70 ratio | Schema | Policy violation (alias) |
|---|---|---|---|---|---|---|---|
| 0.7091 [0.682, 0.736] | 0.7015 [0.658, 0.728] | 0.8117 [0.791, 0.832] | 0.8145 / 0.7933 | 0.0212 [−0.021, 0.063] | 0.376 [0.318, 0.434] | 100% | 0.0804 (0.0448) |

- **Routing:** `needs_human` 34.7%. Action accuracy on the rows not routed is 0.936.
- **Coverage:** action accuracy is 0.955 on the most confident 50%, 0.929 at 70% and 0.862 at 90%.
- **Per noise kind (intent / action):**

  | code_mixed | empty | screenshot | two_intents | typo | very_long |
  |---|---|---|---|---|---|
  | 0.70 / 0.76 | 0.62 / 0.80 | 0.50 / 0.80 | 0.68 / 0.86 | 0.72 / 0.70 | 0.68 / 0.84 |

- **Caveat:** the configuration was chosen by looking at dev results on 13 Oct, so these dev numbers may be slightly optimistic.

### P0-4. Full-dev LLM test (pre-registered)

The pre-registration is `experiments/llm_tail/PREREGISTRATION.md`. It was committed and pushed (`7f4edfb`) before the first API call.

**Runs:**

| Model | Rows | Host | p50 / p95 | Cost |
|---|---|---|---|---|
| GPT-6 Luna `openai/gpt-6-luna-20260922` (OpenAI host pinned, fallbacks off, reasoning none) | 2,300 | OpenAI | 1.5 / 2.8 s | $0.47 billed, $1.46 at list uncached |
| Claude Haiku 5.5 `claude-haiku-5-5` (effort low) | 2,300 | Anthropic | 1.7 / 3.3 s | $0.72 at list with caching, $2.48 uncached |

- **Prompt and examples:** the probe's V3 prompt (SHA-256 `3511a0a0…`), with the same retrieved examples.
- **Failures:** none. One Luna output was invalid JSON, and the fallback kept the CPU answer.
- **Abstentions on labelled rows:** Luna 82, Haiku 156. The fallback keeps the CPU answer.
- **Cumulative spend:** Anthropic $1.16 (cap $3), OpenRouter $1.75 (key shows $3.25 left).

**Primary result** (1,940 fresh rows, paired with the CPU model; 95% intervals from a conversation bootstrap):

| System | Intent acc | Macro-F1 | Action acc | Pre-registered rule |
|---|---|---|---|---|
| A. Luna on the 34.7% the CPU model would route | +0.000 [−0.013, +0.012] | +0.017 [−0.003, +0.037] | +0.014 [−0.001, +0.029] | fails |
| B. Haiku intent, every row | −0.003 [−0.024, +0.018] | +0.031 [+0.004, +0.054] | 0 | passes (macro-F1) |
| C. Both | −0.003 | +0.031 | +0.014 | fails |

All three systems still pass every bar on all 2,300 rows.

**Exploratory, not decisive:** Luna on the least-confident share by CPU confidence, action change.

| Share | 10% | 20% | 30% | 40% | 50% | 100% |
|---|---|---|---|---|---|---|
| Action change | +0.011 | +0.024 [+0.012, +0.035] | +0.019 | +0.010 | +0.005 | −0.015 |

At 20%, the top-70 ratio worsens by 0.07, from 0.37 to 0.43, which is still far under the 0.85 bar. Luna for intent on every row gives macro-F1 +0.028 [+0.003, +0.055], about what Haiku gives.

**Reading:**
- The LLMs and the CPU model reach the same intent accuracy but differ by flow. Haiku is up 11.5 points on manage_account and down 11.4 on shipping_issue.
- B's macro-F1 gain comes mostly from rare intents.
- No pre-registered system moves action by 2 points. Luna at 20% might, but that share was picked after seeing the results.

The decision is in `DECISIONS.md` D2.

### P0-5. Confirmation on train fold B (pre-registered)

- **Pre-registration:** `experiments/llm_confirm/PREREGISTRATION.md`, pushed in `bd6fc45` before any call.
- **Rule chosen on dev first:** 5-fold cross-validation by conversation over six candidate intent combinations (`rule_selection_dev.json`). The best was "LLM intent when CPU p(intent) < 0.95, accepted only if it is in the CPU top 3", at 0.750 out-of-fold with Haiku and 0.747 with Luna, against 0.709 for the CPU model. Switching per flow scored 0.730, and always using Haiku scored 0.712.

**Test rows:** fold B, 3,966 clean rows.
- The CPU model scores 0.745 intent and 0.833 action there.
- Its confidences are out-of-fold. `prep.py` asserts they reproduce the training threshold exactly.
- The LLMs' retrieval index is fold A only.

**Runs:**

| Model | Rows | p50 / p95 | Cost | Notes |
|---|---|---|---|---|
| Luna | all 3,966 | 1.6 / 2.8 s | $0.80 billed | 1 invalid output, 145 abstentions |
| Haiku | the 2,060 gated rows | 1.9 / 3.4 s | $0.68 at list with caching | 215 abstentions |

- **Failed calls:** none.
- **Cumulative spend:** Anthropic $1.84, OpenRouter $2.55 (the key shows $2.45 left).

**Result** (paired with the CPU model; 95% conversation bootstrap):

| Step | Intent | Macro-F1 | Action | Top-70 ratio | Rule |
|---|---|---|---|---|---|
| Luna action, CPU conf < 0.642841 (16.7% of rows) | 0 | 0 | +0.0247 [+0.017, +0.032] | +0.056 (0.324 → 0.380) | passes |
| Luna intent, gated top 3 (51.9% of rows gated) | +0.0144 [+0.004, +0.024] | +0.020 | 0 | 0 | fails (< 0.020) |
| Haiku intent, gated top 3 | +0.0129 [+0.003, +0.023] | +0.022 | 0 | 0 | fails (< 0.020) |

- **Action step cost:** $0.03 per 1,000 messages billed, $0.11 at list uncached.
- **Latency with the action step:** p50 15 ms, p95 about 1.8 s, p99 about 2.7 s. This is an offline estimate: single calls at concurrency 8, not yet a load test.
- **Reading:** dev's 20% share gave +0.024 on dev, and fold B gives +0.025 on rows no choice had seen. The intent combination helps, but by less than the bar fixed in advance.
- **Decision:** `DECISIONS.md` D2.
