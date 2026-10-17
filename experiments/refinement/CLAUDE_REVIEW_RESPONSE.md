# Response to the bounded Claude API critique

The first call exhausted 4,096 output tokens on provider reasoning and returned
no visible critique. It is recorded, but is not review evidence. The second call
explicitly disabled reasoning and returned a complete critique. Both token/cost
records are preserved in `results/refinement-20261017/`.

- Accepted: make LLM-disagreement review explicit, use held-out paraphrases, and
  preserve the distinction between selection and confirmation. The regression
  test reproduced automatic handling after a changed LLM action with zero policy
  thresholds; an explicit route reason fixes it.
- Clarified: the independent flow-head permission mass is an uncertainty signal,
  deliberately distinct from the intent-mapped flow's deterministic check. Neither
  is a true-permission guarantee. Changing both to one guessed flow does not fix
  uncertainty about the actual flow.
- Rejected finding 3 as a demonstrated defect: the state table's class order is
  copied directly from the same predictor producing `pi`; classifier classes all
  come from fitting labels, which also construct the intent-to-flow dictionary.
  The proposed mismatch is worth an invariant check, not an observed bug.
- Rejected finding 8: `gate.clean` already calls `_injection` on both cleaned and
  folded text, and `_injection` includes `_LABEL` for customer turns. No missing
  folded-label check exists.
- No hash gates added. The split is deterministic, its conversation IDs are
  saved, the fitting function's split dependency is directly inspected, and
  confirmation is excluded from fitting. The selection artifact is frozen before
  confirmation. A hash cannot replace those substantive checks.
- Selection calibration still shares temperature/alpha decisions; this is stated
  in the protocol and selection output. Confirmation uses frozen fitted models
  and thresholds. It would be incorrect to refit thresholds on confirmation.

The API critique is advice that was checked against code, not another empirical
validation of predictive quality.
