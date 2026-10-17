# Independent review, round 3 — final code and decision

Reviewed 17 October 2026 against `543cfc8` plus the working-tree refinement, including policy version `2026-10-17.2`. **No blocking production-code defect found.** The action-blend rejection is justified. The additional permission-review rule meets the frozen guard-only decision criteria, with material limits on transfer to the shipped service. This is a code/evidence review, not a declaration that pending final service measurements passed.

## Scope and checks

Read both prior reviews, `PROTOCOL.md`, `REPORT.md`, selection and confirmation JSON, the experiment runner, service evaluation helper, production/policy/generator diff, changed tests and documentation. Traced final-action selection, permission support, confidence capping, policy lookup/reload interaction and decision accounting. Only this review file was changed. No delegated agents, training, full suite, SSH or API calls were used.

Ran `.venv/bin/python -m pytest tests/test_refinement_invariants.py tests/test_refinement_policy_guard.py -q`: **12 passed in 8.82s**. The saved 517-passed/2-skipped log is earlier evidence, not independently verified final-version full-suite evidence. Root owns final Linux/full-suite, CPU HTTP and hybrid HTTP checks and expense reconciliation; their pending status is neither a defect finding nor an assumed pass.

## Code correctness

- `triage/pipeline.py:369–410` checks the final accepted action, including LLM replacements, against the retained flow distribution. The matrix is binary permission with aliases applied, so the dot product is permitted probability mass rather than the action prior's reweighted score. The request retains its policy object across the LLM wait; `_prior(policy)` uses that same object even when another request has loaded a new policy.
- Permission support is an additional review reason. It neither changes the action nor removes off-policy, gate, procedure-rule, second-intent or low-confidence review. Version .2 excludes only `policy_uncertainty` from the pre-threshold cap decision. Existing cap reasons still apply, including when uncertainty coexists with them. A permission-only or permission-plus-low-confidence case preserves the action score, consistent with the controlled prototype.
- Accepted LLM disagreement forces review after mask acceptance, independently of editable thresholds. A rejected masked answer does not become the final action or trigger a false accepted-disagreement count. Agreement remains distinct from proof of correctness.
- `triage/policy.py:169–176` accepts optional tier values in [0,1], rejects unknown tiers, booleans, nonnumbers, nonfinite values and out-of-range values, and defaults missing tiers to zero. The generator and shipped file agree on the new setting/version. Existing load failure handling retains the last valid policy.
- Service decision labels have three bounded values, derive from explicit boolean fields on successful traces, and are counted under the metrics lock after the replay early return. No new customer text is logged.

## Regression-test assessment

The tests exercise real Triage prediction rather than just reimplementing the guard: a high-risk uncertain example routes without changing action, intent or confidence; a low-risk control stays automatic; an empty setting preserves legacy routing; and a high-risk CPU action replaced by a low-risk LLM action uses the final tier. The zero-threshold disagreement test catches the original numerical-coincidence bug. Mask rejection, agreement, replay deduplication and malformed/nondecision traces receive distinct assertions.

Historical threshold tests now explicitly use a guard-disabled Phase 2 policy; the shipped-policy fixtures remain available. This preserves the old claim's scope rather than rewriting old expected numbers to conceal a new route rule. The earlier temporary-mask fixture now restores its file/store.

Nonblocking coverage opportunities: add the opposite final-action case (low-risk CPU to uncertain high-risk LLM), an exact support-boundary case, and a mixed uncertainty-plus-existing-flag cap assertion. Loader tests sample only some rejected types even though the implementation covers the others. Model-dependent cases skip without the bundle and search public dev for suitable examples; ensure the final validation actually runs them with the bundle, as this local run did.

## Experiment validity and decision

The current runner preserves the round-two fitting boundaries: predicted intents and observed state at inference, raw blending before a single prior, grouped selection confidence fits, and frozen configuration for confirmation. The shared temperature/alpha selection caveat remains disclosed. Confirmation resamples whole conversations, with matched multiplicities across contrasts. No new leakage path was found. This review did not inspect the remote fitted artifacts; provenance remains bounded by the saved metadata/code and root's execution record.

Alpha .30 gains 0.005992 accuracy with interval [-0.000657, +0.012480]. It fails both the +.010 point requirement and positive lower bound: retaining the production action model is the correct decision.

The pre-confirmation guard-only clarification resolves round two's main attribution objection. On identical reference actions, calibration and cutoffs, the .90 guard changes forbidden automatic suggestions 37 to 32, high-risk 8 to 3, and reviews 752 to 767. Its +0.4993 review points and -0.1664 action-only cost units/100 pass the frozen point limits. The forbidden-rate interval excludes zero; the action-only cost interval [-1.1091, +0.4995] does not establish cost savings. The combined comparison is retained and must not substitute for this guard-only contrast.

## Material residual risks and reporting work

1. **Transfer to production is incomplete.** Confirmation used a freshly fitted reference and selected cutoffs (high approximately .7899, low .7250); production keeps its original model and .8883/.7104 cutoffs. Confirmation also omits the gate and LLM. The same .90 rule is therefore supported as a bounded review intervention, not a proven production effect size. The intermediate public-dev CPU HTTP run adds four reviews while forbidden counts remain 25 total / three high-risk. Preserve this counterevidence beside the positive controlled contrast. The .2 cap correction cannot itself make this CPU forbidden-count result improve because actions/routes are unchanged.
2. **Confidence and authorization remain distinct.** Confidently wrong flows can escape both checks; three high-risk forbidden suggestions remain in confirmation. Hybrid confidence still uses the unfitted CPU/LLM agreement arithmetic. Explicit disagreement review fixes an operational escape, not calibration, and agreeing wrong answers remain possible.
3. **Evidence is small and previously studied.** Five avoided forbidden suggestions are limited support; the bootstrap reflects this sample, not robustness to new customers, policies or providers. The split is a new holdout within previously used public data. The 7/10 attack-style and 0/10 benign gate diagnostic does not establish universal injection resistance or model-level immunity.
4. **Final report reconciliation remains root work.** Retain capped-run provenance, replace pending-measurement text with verified .2 results, and report actual API spend separately from cached/uncached list-price estimates. At inspection, `refined_cpu/summary.json` still described the intermediate .3824 top-70 result; it must not be attributed to .2 without the final run. A stale historical sentence in `DECISIONS.md` still says agreement is not counted; qualify it as historical or update it. `docs/DESIGN.md` calls off-policy a check of the “top predicted flow”; more precisely it uses the returned intent's mapped flow, with direct-flow fallback.

No code change is requested as a blocker. Complete the already planned final validation and reporting, and retain the narrow claims above; do not retune using confirmation.
