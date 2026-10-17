# Independent review, round 2 — before confirmation

Reviewed 17 October 2026 against baseline `543cfc8` and the current working-tree changes. **Safe to run the frozen confirmation as a CPU blend/routing experiment. No demonstrated leakage or production-code correctness blocker found. This is not deployment approval or confirmation of a gain.**

## Scope and validation

Read the round-one brief/findings, protocol, Claude response, complete state-blend runner, selection artifact, partition membership metadata, production diff and focused tests. Traced dependencies through classical training/prediction, input projection, state extraction, policy evaluation and service accounting. No SSH, API calls, training, confirmation execution, confirmation labels or confirmation results were accessed. No additional agents were spawned.

Ran `.venv/bin/python -m pytest tests/test_refinement_invariants.py -q`: **4 passed in 4.58s**, including the real Triage path for changed action, agreement and mask rejection, plus service replay accounting. The prior 512-passed/2-skipped Linux result is background evidence; this review did not repeat that suite. Root still owns full checks and HTTP validation.

The local `models/refinement-control.joblib` is absent. Consequently this review verifies the fitting code and saved selection/partition metadata, not the contents or provenance of the remote fitted bundle. Root must use the already fitted, selection-producing artifacts for confirmation; do not substitute the submitted model or refit after examining outcomes.

## Experiment mathematics and leakage

- Saved groups are pairwise disjoint: fit 13,993 rows / 5,433 groups; selection 3,003 / 1,165; confirmation 3,004 / 1,165. The two-stage GroupShuffleSplit implementation matches the stated seeds and intended proportions. Proportions are group proportions, with approximately matching row proportions.
- `prepare()` replaces `C.folds` with `(fit, selection)` for the complete training call. All vectorizers, intent/flow classifiers and action regression fit on fit rows. Temperatures and the original stacker use selection only. Confirmation is not passed to any fitting operation. The routine loads the train file to partition it, but confirmation outcomes have no computational path into these fits.
- Transition counts, global action frequencies and intent-to-flow mapping come from fit rows. The hierarchy is global -> last action -> flow/last action -> intent/last action, with fixed strength 5 at each conditional level. Each shrinkage step is a normalized convex combination. Inference sums these conditional distributions using predicted intent probabilities; it does not index by a scored row's gold intent. Class-order assertions address the potential alignment concern.
- State extraction reads observed action-speaker turns through projected inputs. Raw action probabilities and table probabilities are mixed before a single policy-prior application. Gold action and permitted lists in `facts()` are evaluation targets, not inference inputs.
- Alpha .30 wins the fixed four-value grid, with selection accuracy 0.818515 versus 0.815185: ten extra correct rows, approximately +0.00333. This is below the preregistered +0.010 promotion bar and establishes no prospective gain.
- Five-fold GroupKFold correctness scores exclude each scored conversation from its confidence-model fit. Shared temperature fitting and alpha choice make these selection estimates incompletely cross-fitted, as already disclosed. The sealed confirmation fixes the primary validity concern; do not describe selection routing estimates as independent validation.
- Threshold enumeration implements the stated strict `confidence < threshold` rule, including all-review via 1.01. Forced off-policy rows are excluded from threshold optimization and included in reported review/cost. Guard support is the direct flow-head probability mass of flows permitting the selected action, with the policy alias. It is an uncertainty signal, not a legality guarantee.
- The paired bootstrap correctly resamples whole conversations with replacement and retains all their rows and multiplicities. It estimates a row-weighted accuracy/cost contrast with conversation clustering. The action pass uses both the frozen absolute gain bar and a positive lower 95% interval bound.

## Findings that constrain interpretation and promotion

### 1. The routing pass is a combined-system comparison, not a guard-only acceptance test

`run_state_blend.py:228–242` compares reference/no guard with candidate/selected guard. Both action selection and confidence/thresholds differ. Although all four summaries are retained, only this combined contrast gets intervals and a routing pass flag.

**Concrete handling:** name that result as the combined candidate comparison. Do not use `routing_change_pass` to deploy a guard on the unchanged reference if `action_change_pass` fails. If a guard-only rollout is intended, freeze its contrast and acceptance rule before opening confirmation: reference/selected guard versus reference/no guard, with unchanged reference actions, calibration and cuts. Candidate guard versus candidate/no guard can separately describe the guard contribution. Preserve the original combined primary contrast and do not select a favorable contrast after seeing confirmation.

This does not block running the currently specified comparison or retaining it as a negative experiment. It blocks an unsupported guard-only promotion claim.

### 2. These routing figures are not full production-pipeline figures

`data()` uses Predictor sanitization; `summary()` routes by off-policy status, fitted confidence and high-risk permission support. It does not execute `gate.clean`, gate routing/abstentions, configurable rule masks, second-intent routing or the LLM path. The default policy currently has no active rules, but the omitted gate still matters. The production diff also does not install the experimental blend, calibrators or guard.

**Concrete handling:** report this as a controlled CPU selector/routing comparison. Before deployment, integrate the accepted behavior and validate matched inputs through the real service, including gate and mask semantics. Do not label confirmation as an end-to-end HTTP or hybrid-LLM gain. Keep production model defaults unchanged if the action criterion fails.

### 3. Remaining reporting limits are nonblocking for this frozen experiment

- The script does not report Brier/log loss, exact matched-coverage errors, literal unaliased violations, or bootstrap intervals for automatic forbidden-action deltas. It cannot establish all calibration and safety claims requested in round one. Report the available counts and cost assumptions narrowly; supplementary fixed diagnostics must not become new selection criteria.
- The top-70 statistic includes all confidence ties at its cutoff, so coverage can exceed 70%. Report actual coverage or use a fixed tie rule if interpreting it as an exact matched-coverage measure.
- Safety counts can be small. The routing pass uses point-estimate limits, consistent with the current protocol; it is not a statistically significant reduction in forbidden actions merely because the flag is true.
- Selection and confirmation reload the policy from disk, while action prior probabilities come from the fitted bundle. Keep the reviewed policy and fitted artifacts unchanged across these stages. A substantive change requires resolving the mismatch before opening confirmation, not adding administrative hash gates.
- Confidence remains CPU-only. Existing LLM confidence arithmetic and possible overconfidence on agreeing answers remain unresolved; the explicit disagreement rule does not calibrate the hybrid system.

## Production safety and telemetry review

The new `used and action != w.action` condition is evaluated after LLM acceptance/mask rejection. It forces human review for every accepted changed action independently of editable thresholds, retains agreement behavior, and does not incorrectly treat a rejected suggestion as the final action. Existing final policy violations remain checked. The tests demonstrate the zero-threshold case that previously escaped this rule.

The added trace fields distinguish CPU agreement and mask rejection without adding customer text. Service decision labels are bounded to three known values; accounting stays under the existing metrics lock and after the replay early return. The focused test verifies accepted agreement, accepted disagreement and mask rejection are each counted once despite replay. Here “accepted” means adopted as the proposed action, not automatically authorized: an accepted disagreement is explicitly reviewed.

Nonblocking test maintenance: the module-scoped scenario is mutated by the mask test and is not reset afterward. Current order passes; restoring the temporary policy/LLM after each test or using independent fixtures would prevent order-sensitive failures. This is not evidence of a production defect.

## Decision

Proceed with the frozen confirmation using the selection-producing artifacts, retain the preregistered criteria, and report failure without retuning. No confirmation outcome was consulted for this decision. Full-suite, service/HTTP and any later deployment checks remain root's responsibility. A separate guard-only deployment needs the explicit pre-outcome contrast described above.
