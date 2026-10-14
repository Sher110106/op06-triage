# Decisions

"Dev" means the 2,300 labelled published rows (2,000 dev plus 300 dev_noisy); "fold B" means 3,966 train rows held
out from fitting the intent, flow and action models (fold B did fit the confidence model, the temperatures and the
routing thresholds; D2 item 3). Dev helped choose the system, so the hidden test will score lower. Log ids (R1, P0-4, …)
are entries in `EXPERIMENT_LOG.md`, which marks the R-entries as reconstructed. Run names are folders under
`results/raw/`.

## D1. The 9% of gold actions outside the written procedure

**Initial hypothesis (reconstructed).** The written procedure is the rule. Masking every action the predicted flow
does not permit would bring violations near zero at a small cost in accuracy. The 14 Oct draft of this file still
said a hard mask gives near-zero violations.

**Options.**
1. **Follow the label:** learn gold actions with no policy input.
2. **Hard mask:** allow only actions the predicted flow permits.
3. **Soft prior:** weight each action by its chance of being permitted under the flow probabilities; read the
   procedure from a versioned file; treat `membership-privileges` and `membership` as one action; send any
   off-policy answer to a person.

**The constraint that mattered.** The grader scores the action against what agents did, and agents broke the
procedure on 8.8% of dev rows (5.7% once the two membership names count as one; R1, `report_official.json`). The
manager asked for what the agent is *allowed* to do. Enforcing the procedure costs accuracy on exactly the rows
where agents deviate, so the service must show that choice, not hide it. She must also be able to change a rule
without touching code.

**What changed my view.** One-pass ablations on dev, LLM step off (`experiments/ablations/ablations.md`;
exploratory, not pre-registered):

| Run | Action Δ [95% CI] | Violation (with alias) | Routed |
|---|---|---|---|
| Shipped: soft prior + alias (`dev-pipeline-cpu`) | – | 0.0804 (0.0448) | 28.2% |
| `ablation-hard-mask` | −0.010 [−0.018, −0.004] | 0.0743 (0.0383) | 26.8% |
| `ablation-no-alias` | −0.026 [−0.034, −0.018] | 0.0452 (0.0435) | 29.6% |
| `ablation-no-prior` | −0.012 [−0.020, −0.004] | 0.1200 (0.0848) | 34.0% |
| Gold's own rate | | 0.0883 (0.057) | |

- **The hard mask does not reach zero.** It masks by the *predicted* flow; the grader checks the *gold* flow. When
  the flow guess is wrong, the mask can drop the right action and keep one the gold flow forbids. The 13 Oct
  probe's version scored 0.0774 (`results/audit/AUDIT.md` §2).
- **Much of the gap is a naming mismatch.** In R1, 50 baseline errors were off-policy only because the lists say
  `membership-privileges` where the data says `membership`. Without the alias, 86 dev rows change: 70 were right
  before, 11 after. The literal violation rate halves only by refusing what agents do.
- **No option fixes real deviations.** Errors whose gold action breaks the procedure: 94 in the baseline, 72 in v1
  with the LLM step off (clean dev, `experiments/taxonomy/taxonomy.md`).
- **Off-policy answers are often wrong:** 63% right on fold B (134 rows, 3.4%), 57% on dev (95 rows, 4.1%; P2-1,
  `experiments/routing/results_routing.json`). Routing them costs 0.55 more per 100 messages on fold B (40.49
  against 39.94, where a review costs 1 and a wrong action 10 or 3) and lifts the routed share from 19.1% to 20.5%.

**Choice.** Option 3. The procedure and the alias live in `policy/policy.toml` (version `2026-10-14.2`). The
service reloads it without a restart, and `scripts/replay_policy.py` re-scores dev before a change ships.
Off-policy answers go to a person as a policy rule, not a cost choice: the cheapest fold-B configuration had this
routing off. The submitted run (`dev-http-llm`) violates at 0.0839 (0.0443 with the alias), under gold's rate.

**What would make me reverse it.**
- If the manager rates an off-procedure suggestion worse than a wrong one for some action, such as refunds: mask
  that action alone with a rule in the policy file. A full hard mask costs about a point of action accuracy for
  0.006 fewer violations.
- If reviewers keep approving the off-policy answers they receive, the agents' habit is the real procedure: add
  those actions to the flow lists rather than stop routing.
- For a new rule, I start with `route`, not `mask`: the service sees only 8 turns, so it may miss the earlier
  step a rule needs. In a 14 Oct replay of "a refund needs identity or purchase checked first" (2,300 rows, LLM
  step off), `route` changed no action and routed 8 more rows. `mask` changed 39 actions (26 right to wrong, 4
  wrong to right), took action accuracy from 0.8117 to 0.8022 and raised violations from 0.0804 to 0.0830
  (`results/raw/policy-replay-refund-{route,mask}/report.txt`; policies in `experiments/policy_replay/`; rerun
  with `.venv/bin/python experiments/policy_replay/run.py`; exploratory).

## D2. Does an LLM belong in v1?

**Initial hypothesis.** No. In the 13 Oct probe (R9, 360 rows) no LLM beat the CPU model on the next action (Luna
came closest, −0.022 [−0.064, +0.017]), and the intent gains sat on `very_long` rows, which de-duplication already
fixes. The first pre-registration said v1 would ship with no LLM if nothing passed.

**Options.** Pre-registered in `experiments/llm_tail/PREREGISTRATION.md`, pushed 8 s before the first API call
(`AUDIT.md` §4): **none**; **A**, GPT-6 Luna answers the 34.7% of rows the CPU model would route; **B**, Claude
Haiku 5.5 picks the intent on every row; **C**, A and B. Later tests added Luna on the least-confident 20% and a
CPU-only encoder blend.

**The constraint that mattered.** The action is what the manager acts on, and an LLM adds cost, a p95 near 2 s and
a provider that can fail. Each test fixed its bar in advance: a gain of at least 0.020 with the interval above
zero. Any failed call must leave the CPU answer.

**What happened, including what failed or stayed unclear.**

1. **The dev test passed an option I did not ship** (P0-4; runs `llm-tail-*`; `experiments/llm_tail/results.json`;
   1,940 rows the probe never saw). A failed: action +0.0135 [−0.0011, +0.0286]. **B passed on macro-F1 alone,
   +0.031 [+0.004, +0.054]**; `results.json` records `adopt_B: true`. I did not ship B: intent accuracy moved
   −0.003 [−0.024, +0.018], the gain came from rare intents, B cannot change the action, noisy-row intent fell 3.9
   points on the fresh rows (`AUDIT.md` §2; no script in the repo stores this analysis), and it calls Haiku on
   every row ($1.08 per 1,000 messages uncached, p95 3.3 s per call). Commit `012c85d` recommended against B at
   the time, for the macro-F1-only gain, the unchanged action, and the cost and latency, and left the call open;
   it gave the noisy fall as 1.3 points over all 2,300 rows, and the 3.9-point fresh-row figure came later from
   the audit. I overrode my own pre-registered rule.
2. **A lead found by looking.** Luna on the least-confident 20% gave action +0.024 [+0.012, +0.035] on dev. I picked
   that share after seeing results, so it needed fresh rows.
3. **Fold-B confirmation** (P0-5; `experiments/llm_confirm/PREREGISTRATION.md`, pushed about 1 s before the first
   call; runs `llm-confirm-*`; `results_B.json`). No LLM choice had used fold B, though it had fit the confidence
   model, the temperatures and the 0.8498 threshold. Luna's action below CPU confidence 0.642841 (16.7% of rows):
   **+0.0247 [+0.017, +0.032], pass.** Intent steps by accuracy: Luna +0.0144 [+0.0042, +0.0242], Haiku +0.0129
   [+0.0025, +0.0227], both under the bar.
4. **The intent bar narrowed between tests.** The first test accepted accuracy *or* macro-F1; the fold-B test,
   accuracy only. Under the first test's macro-F1 criterion both fold-B intent steps pass: Luna +0.0202
   [+0.0072, +0.0323], Haiku +0.0219 [+0.0081, +0.0371]. My reason, reconstructed because the pre-registration
   gives none: the lead under test was an accuracy lead, picked by out-of-fold accuracy (0.709 to 0.750,
   `rule_selection_dev.json`), and I had just declined B because a macro-F1 gain with flat accuracy did not pay for
   an LLM call. Under the bar written first, an intent step would have shipped. Read the result as "fails the
   second test's bar", not "no intent gain"; both accuracy intervals sit above zero.
5. **The encoder near miss** (`experiments/encoder/PREREGISTRATION.md`, commit `d1f7258`; `results_encoder.json`;
   run `encoder-ett32A`). A blend of the classical model and Ettin-32M fine-tuned on fold A (action 0.5/0.5, intent
   0.3/0.7, weights chosen on dev) beat the classical model by +0.0273 [+0.0203, +0.0341] action on fold B, a real CPU gain. Against the
   shipped Luna step: +0.0026 [−0.0053, +0.0098], missing the −0.005 floor set in advance by 0.0003. By the rule it
   stays out of v1. Caveats: training started about 2 minutes before the pre-registration commit (`AUDIT.md` §4),
   and the run did not record the Hugging Face revision.
6. **Two live runs of the shipped service** over HTTP on an Apple M1, concurrency 8, 2,309 rows:

   | | `dev-http-llm` (submitted) | `dev-http-llm-r1` (first) |
   |---|---|---|
   | Action | 0.8365 [0.8162, 0.8563] | 0.8365 [0.8149, 0.8569] |
   | Noisy gap | 0.0535 | 0.0228 |
   | p50 / p95 | 8.9 ms / 1.913 s | 8.9 ms / 1.656 s |
   | Luna calls | 460 (19.9%): 418 ok, 37 abstained, 5 timed out | 460 |
   | Cost per 1,000 messages | $0.125 uncached, $0.040 with caching | $0.040 with caching |

   71 of 2,300 actions differ, 21 right each way, and the noisy gap moved 3 points: Luna's answers varied between
   runs. The request sets no temperature or seed, so the provider's default sampling applies; a temperature-0
   rerun of the 460 gated rows (about $0.29) was not tried. `tests/test_prompts.py` checks that the service's
   prompt equals the tested one, and the examples changed for only 2 of the 460 called rows (P3-4). Code in
   `triage/pipeline.py`, `service.py` and `llm_action.py` also changed between the runs, unrecorded, so the 71 rows
   are not a clean measure of Luna alone (P3-5). All 6,276 Luna responses in the ledger (`results/raw/llm-ledger/`)
   returned the id `openai/gpt-6-luna`, though I requested `openai/gpt-6-luna-20260922`. The live runs' service
   log did not record the returned id, so I cannot show which snapshot answered them; the log now records it
   (P5-2). Because dev set the 0.642841 gate, fold B's +0.025 is the cleaner estimate.
7. **Which rows gain.** A changed action gets confidence (CPU confidence + 0) / 2, under gate / 2 (now 0.3214),
   below both routing points. So every action Luna changes goes to a person: 216 of 216 in the submitted run
   (against `dev-pipeline-cpu`), Luna right on 100 and the CPU model on 43. When Luna agrees, confidence rises: the
   routed share falls from 28.2% to 24.3%, and accuracy on rows that go out alone from 0.920 to 0.915 (P2-1,
   `dev-http-llm/report.json`).

8. **The step costs noisy-row margin.** The noisy gap is 0.0212 with the step off, 0.0338 replaying recorded
   answers, and 0.0228 and 0.0535 in the live runs; the submitted run's interval reaches 0.0972 against the 0.10
   bar. On noisy rows the step gains nothing (0.79 against 0.7933 off), and the top-70 ratio worsens from 0.376
   to 0.433. I weighed these against the clean-row gain (0.8145 to 0.8435) and the lower routed share, and kept
   the step; it is the closest call in this decision.

With the step off, dev action is 0.8117 (`dev-pipeline-cpu`) and p95 is 38 ms on the M1 (`dev-http-cpu`). Both
configurations pass every bar on dev, so the bars do not decide this.

**Choice.** v1 ships Luna's next action below CPU confidence 0.642841, behind a config flag, with a 4 s deadline, a
circuit breaker and the CPU answer as fallback. Calls go through OpenRouter to the OpenAI host, fallbacks and
reasoning off. A gateway other than OpenRouter gets OpenAI's field names in place of OpenRouter's; that form has
run only against a local fake (P5-2). No LLM intent step; no encoder.

**What would make me reverse it.**
- Fresh-row evidence that the action gain is under 0.02 or its interval includes zero.
- A latency budget that cannot take a p95 near 2 s. Then gate fewer rows (the 10% gate kept +0.017 [+0.011,
  +0.022] on fold B with 8.1% of rows calling; `experiments/ablations/llm_share_fold_b.json`) or turn the flag off.
- The encoder blend, with its confidence model and thresholds refit, matching the Luna step on fresh rows within
  the floor. The rule does not name it the v2 path, but it removes every outside call, so I would try it first.
- A CPU combination of the old (intent, last action) lookup and the learned action model, not tested. Either is
  right on clean dev 86.9% of the time, against 81.5% for the learned model alone, and 35 of the 37 "state ignored"
  rows Luna fixed were already right under the lookup (the same counts in both live runs; `taxonomy.json`
  `rule_vs_learned_and_luna`). If such a combination matched the Luna step on fold B, it would replace it.
- A hidden-test or production noisy gap near 0.10. Then skip Luna on rows with noise signs, or turn it off.
- A fall in Luna/CPU agreement (47.4% on dev, `experiments/chaos/chaos.md`), the first sign of a different model
  behind the alias. The service does not count agreement yet.
- A change to the gate or the high-risk routing point that breaks (gate + 1) / 2 < high-risk point (now 0.8214
  against 0.8883), or gate / 2 < both points. Then an agreeing LLM could send a high-risk action out alone, or a
  changed action could skip review. Nothing in the code checks this (`RUNBOOK.md`, Configuration).
- Reviewers ignoring the suggestion on routed rows. Then the step buys only the lower routed share.

Provider faults do not reverse it: in fake-provider runs, 1,785 of 1,785 rows with a failed call kept the CPU
answer (`chaos.md`, exploratory).

## Smaller decisions

- **Routing points by cost per risk tier** (P2-1, `results_routing.json`). A review costs 1, a wrong action 10
  (refunds, order and account changes) or 3. Fold B picks 0.8883 and 0.7104 and routes 20.5% at 40.49 per 100,
  against 31.1% and 44.58 for v0's single 0.8498. Other prices move the points (5/3 routes 16.4%, 20/10 routes
  40.2%; `policy/policy.toml` lists five).
- **Second-intent flag logged, not routed** (P2-1). Routing on it raised fold-B cost to 44.5 per 100; flagged rows
  are 87% right. On dev it fires on 19 of the 50 `two_intents` rows, 95% right on action.
- **Ship the trained model file** (P3-1, `linux-retrained-cpu`). Retraining on Linux changed 5 of 2,309 actions
  (0.8104 against 0.8117): the solver stops at its 300-iteration cap, so the result depends on the BLAS library.
  `reproduce.sh train` retrains beside the 39 MB file and reports the difference. Raising the cap would change the
  model behind every pre-registered result.
- **One dedicated CPU thread** (P3-2, `linux-threads`). On the two-socket Linux box, CPU work spread over 2 to 8
  threads ran at 39–41 req/s, against 126–137 on one dedicated thread; on the M1 the choice makes no difference
  (304–318 against 307–319 req/s).
- **Stable tie-breaking in retrieval** (P3-4, `triage/retrieval.py`). The default sort broke exact ties differently
  on x86 and ARM, so Luna saw different examples for 3 of 2,300 dev rows on Linux. Commit `3a33423` records
  identical examples on all 2,300 rows after the fix; no raw file holds that check.
- **No LLM intent step** (D2): +0.014 and +0.013 fold-B intent accuracy, under the bar, no action change, and an
  LLM call on 52% of rows.

## Coding-agent assistance

Coding agents (Claude Opus 5.5) wrote most of the code, tests, analysis scripts and document drafts; every commit
carries a Co-Authored-By line. Agent workflows built the v1 service from `docs/DESIGN.md` (P1-1, commit
`fe221d1`). A four-part review raised 24 findings; a second agent tried to refute each, 23 held, and each fix came
with a test that failed before it. Agents proposed options; I made the calls in D1 and D2.

I ran or read myself the full test suite, the smoke run and parity with v0 (P1-1). Agents ran the other checks
below, the audit included (a separate agent session). A check counts as independent when it rests on separate code, a separate machine,
or a file frozen before the result existed.
- **Tests:** `scripts/reproduce.sh test`, 464 tests at `fe221d1`; 514 on the final working tree, all passing on
  the M1 (P5-2). On Linux 6 failed: 4 from the retrained model
  (P3-1), 1 from retrieval tie order (P3-4), and 1 wall-clock timing test, since changed to check the cap.
- **Parity:** with the LLM step off, v1 gives the same intent and action as `dev-classical-v0` on all 2,300 rows
  (`dev-pipeline-cpu-e2e`, P1-1); the published baseline reproduces exactly (P0-2).
- **Audit** (`results/audit/AUDIT.md`): its own scorer written from the problem statement, spend recomputed from
  ledger token counts, a leak test with gold fields scrambled (2,309 identical answers), and byte-identical reruns.
  It found errors in the earlier draft of this file, among them the hard-mask rate and the unstated B override and
  bar change; this version fixes them.
- **Pre-registration timing:** commit and push times against the first ledger call (`AUDIT.md` §4); no
  PREREGISTRATION.md changed after its commit.
- **Two live runs** with the real Luna step, so the run-to-run spread is measured.
- **Linux reproduction** from `git archive` (P3-1): the baseline matches, and the shipped model file gives the same
  labels over HTTP, confidences within 1e-6.
