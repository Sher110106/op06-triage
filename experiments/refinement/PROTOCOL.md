# Refinement review, 17 October 2026

This is a new comparison, not a replacement for the saved submitted results.
The public dataset has been examined in earlier work. The confirmation partition
below is excluded from this round's fitting and selection; it is not a previously
unseen hidden test.

## Questions fixed before running the comparison

1. Does a support-smoothed transition table complement the word/state action
   classifier when its input intent is predicted rather than supplied by an oracle?
2. Can permission uncertainty reduce automatic suggestions disallowed by the
   labelled flow, at a measured review cost?
3. Does the service always route LLM-changed actions even if policy thresholds are
   edited? Previous behavior depended on the relationship between two thresholds.

## Partitions and fitting

Use only `train.jsonl.gz` to fit/select/confirm. Split by conversation ID:
70% fit, 15% selection/calibration, 15% confirmation. GroupShuffleSplit seeds
20261017 and 20261018. Save group membership before fitting. No confirmation
outcome is used until a selection file has been written.

Refit the published classical recipe on fit conversations. Fit temperatures on
selection. Fit transition counts on fit only, with additive hierarchical
shrinkage strength 5 through global -> last action -> flow/last action ->
intent/last action. At inference marginalize the last distribution over predicted
intent probabilities; never use an evaluation row's gold intent or future action.
Mix raw action probabilities with that distribution, then apply the same policy
prior. Select transition weights from {0, .15, .30, .50} by selection action
accuracy, ties favoring the smaller weight.

Refit the eight-feature action-correctness model on selection for each chosen
action distribution. Use conversation-grouped 5-fold out-of-fold correctness
scores to choose risk thresholds. Alpha selection and temperature fitting share
selection rows, so OOF selection routing numbers can still be mildly optimistic;
confirmation routing uses only the frozen fitted models and thresholds.

Explore a permission-support review guard for high-risk actions only: no extra
guard, or require the predicted flow distribution to assign at least .80, .90,
or .95 probability to flows allowing the action. Select with review cost 1,
wrong automatic action costs 10/3, and an additional cost 10 for an automatic
suggestion outside the gold flow's permitted list (membership alias applied).
Report both original action-only cost and this policy-inclusive assumption.
Guard selection does not change action labels.

## Confirmation and stopping

Compare the frozen selected action blend with the freshly trained reference on
the same confirmation conversations. Use 2,000 paired conversation-bootstrap
draws. Promote an action change only for at least +.010 absolute action accuracy
and a positive lower 95% confidence limit; report smaller or uncertain gains as
such. This lower threshold than the earlier LLM +.020 requirement is intentional:
the candidate has no API cost, additional model dependency, or network latency.
Routing changes additionally need no more than 2 review-cost units per 100
regression in original action-only cost, fewer policy violations among automatic
answers, and no more than five percentage points extra review. No retuning after
confirmation outcomes. An unsuccessful action candidate is retained as a negative
experiment, and the shipped action model remains the default.

Public dev/noisy/hostile files are subsequent regression diagnostics only, not
new independent confirmation. Production edits require focused regression tests,
the full existing suite, and matched HTTP latency checks. Do not overwrite saved
submission runs or silently adopt a changed provider model.

## Pre-confirmation clarification after Astra round 2

Written before opening confirmation: the combined candidate comparison cannot
justify adding a guard to the unchanged reference if the action blend fails.
Also freeze a guard-only contrast: reference with its selected .90 guard versus
the same reference without that guard, identical actions/calibration/cutoffs.
Use the same cost/review/forbidden-count point-estimate limits above and report
paired conversation-bootstrap intervals for cost, review and forbidden-action
rate differences. Keep the original combined contrast. Small safety counts and
passing point limits alone do not establish a statistically significant benefit.
These are controlled CPU-routing experiments, not full HTTP/LLM measurements.

## Resources and execution

Root owns heavy jobs, SSH and API spend; only one heavy process at a time.
Linux host: sher@10.1.18.190. API spend ceiling is $10. Credentials stay outside
the repository. A bounded Claude API critique and any predictive API experiments
save request/response provenance and token cost without credentials.
