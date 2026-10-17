# Why the frozen neural candidate failed its operating guard

The ensemble retained both reference high-risk forbidden automatic suggestions and added three more. All three additions kept the reference's action and intent; they changed from reviewed to automatic under the ensemble's separately fitted confidence model and thresholds. The added rows belong to two conversations. This is a posthoc description of the frozen comparison, not another candidate-selection or threshold-tuning step.

On 3,030 confirmation rows, action accuracy increased from 80.5941% to 84.2574%, a gain of 3.6634 percentage points with a paired conversation-bootstrap 95% interval of [2.7610, 4.5857] points. Reviews fell from 853 to 699, and assumed action/review cost fell from 46.9967 to 43.2013 units per 100 rows. Total forbidden automatic suggestions stayed at 34. Their high-risk subset increased from 2 to 5; the remaining subset consequently fell from 32 to 29.

The predeclared operating rule required no increase in the high-risk forbidden-automatic rate. Its observed increase was 3/3,030, or 0.09901 percentage points. The saved paired 95% interval is [0, 0.26552] points. This fails the point guard despite the accuracy and cost improvements. It does not establish a statistically significant population increase in unsafe behavior. The current default is retained.

## What the lower modeled cost contains

Counting directly from the saved predictions, with the policy risk tier of the suggested action and errors measured against the gold action, gives:

| Frozen outcome | Reference | Ensemble | Ensemble minus reference |
|---|---:|---:|---:|
| Reviews | 853 | 699 | −154 |
| Wrong automatic low-risk actions | 187 | 160 | −27 |
| Wrong automatic high-risk actions | 1 | 13 | +12 |
| All wrong automatic actions | 188 | 173 | −15 |
| Total cost at review/low-risk error/high-risk error = 1/3/10 | 1,424 | 1,309 | −115 |

The costs are exactly `853 + 187×3 + 1×10 = 1,424` and `699 + 160×3 + 13×10 = 1,309`. Fewer wrong automatic actions overall therefore coexist with more wrong high-risk automatic actions. This breakdown was verified row by row using `policy.risk` after canonicalizing action names; it was not inferred from rounded aggregate costs.

Action errors and forbidden actions are different measurements. The reference's one wrong high-risk automatic action is permitted under its gold flow. Of the ensemble's 13, eleven are permitted under their gold flows and two are forbidden. Conversely, a suggestion matching the gold action can still violate the labelled-flow permission metric, as the case analysis below shows. The 1/3/10 cost assigns penalties to action-label errors and review; it contains no separate forbidden-action penalty.

For a **posthoc valuation sensitivity calculation only**, keep every prediction and review decision fixed, keep review cost 1 and low-risk error cost 3, and call the high-risk error cost `H`. The ensemble-minus-reference cost difference per 100 rows is:

`100 × (−154 − 27×3 + 12H) / 3,030 = 100 × (−235 + 12H) / 3,030`.

It is zero at `H = 235/12 = 19.5833…`. At `H = 20`, the totals become 1,434 for the reference and 1,439 for the ensemble, so the ensemble costs 0.16502 more units per 100 rows. Thus its cost advantage depends on how the business values high-risk errors. This is not a newly selected cost model, a retuned routing policy, or another adoption criterion: no confidence cutoff or decision changed, and the original frozen guard failure remains the deployment result.

## The five cases

Here, “forbidden” means the suggested action is outside the recorded gold flow's permitted list, with the policy alias applied. It does not establish actual authorization. `human=true` means the suggestion was sent for review.

| Case ID | Gold flow | Gold action | Reference: action / confidence / human | Ensemble: action / confidence / human | Change |
|---|---|---|---|---|---|
| `1224-25` | `single_item_query` | `membership` | `promo-code` / .862743 / true | `promo-code` / .899098 / false | Added automatic case; action unchanged and differs from gold |
| `4821-30` | `single_item_query` | `make-purchase` | `make-purchase` / .954771 / false | `make-purchase` / .956017 / false | Retained; action matches gold |
| `5176-16` | `storewide_query` | `enter-details` | `update-account` / .728823 / true | `update-account` / .841661 / false | Added automatic case; action unchanged and differs from gold |
| `5176-17` | `storewide_query` | `update-account` | `update-account` / .742466 / true | `update-account` / .888968 / false | Added automatic case; action unchanged and matches gold |
| `8227-15` | `purchase_dispute` | `update-account` | `update-account` / .912418 / false | `update-account` / .937811 / false | Retained; action matches gold |

The two reference cases are exactly `4821-30` and `8227-15`. The three additions are `1224-25`, `5176-16`, and `5176-17`. Thus five rows represent four conversations, and the three-row increase represents two conversations. No high-risk forbidden reference case disappeared.

All five pairs also retain the same predicted intent: `out_of_stock_general`, `out_of_stock_one_item`, `manage_pay_bill`, `manage_pay_bill`, and `status_service_added`, respectively, in table order. The failure on these rows is a change in automatic handling, not a newly chosen action or intent.

## Why the permission-support guard did not route them

Frozen high-risk correctness cutoffs are .897663 for the reference and .799597 for the ensemble. Each added case was below its reference cutoff and above its ensemble cutoff. Confidence increased in all three cases, while the ensemble's fitted cutoff was lower. These are different fitted confidence functions, so the numbers should not be treated as interchangeable probability scales or used to attribute the change solely to the cutoff.

The independent permission-support guard remained .90 for high-risk actions. Support is the sum of predicted flow probabilities over flows permitting the selected action. It is not the probability assigned to the recorded gold flow. The frozen ensemble uses a .5/.5 classical/neural flow blend after neural temperature calibration.

| Case ID | Top predicted flow and intent-mapped flow, both systems | Reference permission support | Ensemble permission support | Ensemble probability on gold flow |
|---|---|---:|---:|---:|
| `1224-25` | `purchase_dispute` | .938332 | .961628 | .005581 |
| `4821-30` | `purchase_dispute` | .941581 | .929715 | .032888 |
| `5176-16` | `subscription_inquiry` | .998801 | .988797 | .003458 |
| `5176-17` | `subscription_inquiry` | .995025 | .986694 | .001079 |
| `8227-15` | `manage_account` | .992765 | .976676 | .002493 |

Every selected action is permitted under its predicted-intent flow, so that check does not flag it. Every permission-support value also exceeds .90, so the uncertainty guard does not flag it. Both systems place most probability on flows inconsistent with the recorded gold flow. This illustrates the limitation of predicted permission checks: agreement between a selected action and inferred flows can coexist with a violation of the labelled-flow metric. The current policy has no active prerequisite-mask rules.

## Annotation inconsistency limits the interpretation

For **all five rows, the recorded gold action itself is outside the row's recorded permitted-actions list**, including after the membership alias is applied. This is not a discrepancy introduced by the policy file: for these rows, the policy table and the row-level permission list agree on membership for all 30 model action labels.

Three ensemble suggestions match the gold action exactly and still count as forbidden: `4821-30`, `5176-17`, and `8227-15`. The other two are action-label errors as well as labelled-flow violations. The data therefore contains a conflict between action targets and flow/permission annotations in this set. This analysis cannot determine which annotation reflects the intended real permission. An action-correctness model is trained against the action target, not against actual authorization; those objectives can disagree here.

The annotation conflict qualifies the meaning of the observed count. It does not justify removing cases, changing the metric, relaxing the frozen guard, or promoting the candidate after seeing confirmation. Any public summary should call these “high-risk automatic suggestions forbidden under the labelled-flow metric,” not proven unauthorized executions. No action was executed.

## Evidence and scope

Counts, confidence values, decisions, and intervals come from the saved [confirmation report](../../results/model-exploration-20261017/modernbert_joint/confirmation.json), [reference predictions](../../results/model-exploration-20261017/reference_confirmation.jsonl), and [ensemble predictions](../../results/model-exploration-20261017/ensemble_confirmation.jsonl). Cutoffs, temperatures, and blend weights come from [frozen settings](../../results/model-exploration-20261017/modernbert_joint/frozen.json).

Permission support was recomputed from the saved reference probabilities and neural logits using the frozen settings, `classical.softmax`, and the actual `Policy.prior_matrix`. Both probability exports' IDs were checked against the confirmation partition order. Gold labels and row permission lists were joined by those IDs. Support values above are rounded for display; no thresholds or model settings were changed.

This file contains private case identifiers and no dialogue text. The analysis used saved arrays and predictions only: no model inference, fitting, provider request, alternative configuration, or new confirmation selection. Small counts and repeated rows within a conversation limit generalization. The bootstrap intervals are the existing grouped intervals, not a new independence assumption for the five rows.
