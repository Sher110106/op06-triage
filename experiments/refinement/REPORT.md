# Pipeline review and refinement, 17 October 2026

The action model is retained. The tested transition-table blend did not establish
the required improvement. The service now explicitly reviews every accepted LLM
action change, reviews high-risk suggestions with weak inferred policy support,
and records LLM agreement and mask rejection in metrics. A permission check based
on an inferred flow remains an estimate, not proof of actual permission.

## Review and validity

Astra independently reviewed the complete training/inference/service flow in
[round 1](ASTRA_REVIEW_ROUND1.md), then reviewed the experiment and production
changes [before confirmation](ASTRA_REVIEW_ROUND2.md). A bounded Claude API critique
was checked against code; [the response](CLAUDE_REVIEW_RESPONSE.md) separates valid
findings from incorrect suggestions. Review agreement is not empirical evidence.

The initial Linux baseline suite passed 512 tests, with two retrieval-index tests
skipped. Original submitted prediction files, model weights and raw runs are
preserved. New results live in `results/refinement-20261017/`.

We retrained the complete classical recipe after a new conversation-group split
of the published training file: 13,993 fitting rows, 3,003 selection/calibration,
and 3,004 confirmation. Temperatures, confidence models, transition counts,
blend weight and review settings excluded confirmation. Selection correctness
predictions used grouped OOF fits, although shared temperature and blend selection
still make selection estimates optimistic. This dataset has been used in earlier
research; the comparison is a newly held-out re-split, not a historically pristine
dataset or the hidden test. See [the protocol](PROTOCOL.md).

## Experiment 1: smoothed state transitions

A support-smoothed transition table used fitting labels only. Inference
marginalized over predicted intent probabilities and observed prior actions.
Raw action probabilities were blended before one policy-prior application.
Selection chose transition weight .30 from {0, .15, .30, .50}.

On the 3,004 confirmation rows:

| Measurement | Fresh reference | Frozen blend |
|---|---:|---:|
| Action accuracy | 81.0919% | 81.6911% |
| Action gain | | +0.5992 percentage points |
| Paired 95% conversation-bootstrap interval | | −0.0657 to +1.2480 points |

The rule required at least +1 point and a positive lower interval endpoint.
**Rejected for deployment.** No retuning on confirmation. The frequently cited
86.9% rule/classifier union was an oracle diagnostic involving gold intent, not
deployable blend performance.

## Experiment 2: permission-uncertainty review

The guard uses the independent flow classifier's total probability on flows
permitting the final selected action. Selection chose .90 for high-risk actions.
Before opening confirmation, Astra identified that the combined blend/guard
contrast could not establish a guard-only result. We explicitly froze a second
contrast: identical reference actions, confidence and cutoffs, with/without its
selected guard. The original combined comparison is also preserved.

| Confirmation measurement | Without guard | With guard |
|---|---:|---:|
| Automatic suggestions forbidden by labelled flow, alias applied | 37 | 32 |
| Of those, high-risk | 8 | 3 |
| Cases sent for review | 752 / 3,004 | 767 / 3,004 |
| Action-only cost units per 100 cases | 47.7031 | 47.5366 |

The guard adds 15 reviews, or 0.4993 percentage points. The 95% conversation-
bootstrap interval for the change in forbidden-automatic rate is −0.3656 to
−0.0330 percentage points. Action-only cost change is −0.1664 units/100, with
interval [−1.1091, +0.4995]: no statistically established cost saving. Cost assumes
one review costs 1 and wrong automatic high/low-risk actions cost 10/3. An extra
penalty of 10 for a forbidden automatic suggestion helped select the guard;
both original and policy-inclusive cost are reported.

The pre-specified guard-only point limits passed: fewer forbidden automatic
suggestions, no more than five points extra review, and no more than two units
extra action-only cost/100. **Adopted as an additional review rule, not an action
replacement or legality guarantee.** Three high-risk forbidden suggestions
remain in confirmation; confidently wrong flow predictions can still escape.

## Service changes

- `policy/policy.toml` version `2026-10-17.2` sets
  `permission_review = { high = 0.90 }`. Missing tiers or zero disable the extra
  guard. Invalid bounds/tiers are rejected; reload keeps the last valid policy.
- Policy support is checked after LLM acceptance, on the final action. It cannot
  raise confidence or disable any other review reason. `policy_uncertainty`
  alone preserves action-correctness confidence while forcing review.
- `llm_disagreement` forces review whenever an accepted LLM changes the CPU action,
  even if both confidence thresholds are zero. The old threshold-dependent escape
  was reproduced before the fix. Agreeing and masked/rejected answers retain their
  respective behavior.
- Traces/logs distinguish agreement, disagreement and masked rejection.
  `/metrics` reports `llm.decisions`; cache replays do not double count decisions.
  Existing privacy rules exclude customer/prompt/provider-error text.

## Additional diagnostics

A gate-only probe covered ten new instruction/role/review-override phrasings and
ten curated benign controls. Seven attack-style cases routed; three did not.
No benign control routed, and customer text never became an observed action in
all 20 cases. This is a small diagnostic, not a population recall/false-positive
estimate; an unflagged phrase is not proof of model compromise. We did not expand
regexes and claim general injection immunity.

A three-row real-provider HTTP smoke produced three successful Luna calls, two
accepted agreements and one reviewed disagreement. A replay made no new call.
It is compatibility/telemetry evidence, not an accuracy estimate. Returned model
and provider IDs are recorded; a dated requested model can still return an alias.

## Implementation iteration: keep confidence separate from permission review

The initial service guard reused the existing flag confidence cap. On public dev,
this changed CPU top-70 error ratio from .3760 to .3824 despite identical actions.
It also differed from the controlled confirmation, which never capped confidence
for this guard. A regression test reproduced confidence dropping from .701195 to
.5 with `policy_uncertainty` as the only review reason.

Version `2026-10-17.2` preserves action-correctness confidence for that reason
alone; other existing cap reasons keep their behavior. Review and action choices
are unchanged. The earlier capped CPU/hybrid runs are retained explicitly as
intermediate evidence, rather than selected or silently overwritten.

## Final service validation

The Linux retrieval index was rebuilt; both retrieval-index tests that had been
skipped earlier then passed. The final Linux suite passed **529 tests with no
skips** in 153.33 seconds (`results/refinement-20261017/op06-completion-full-tests.log`).
Retrieval rebuild and test evidence is in
[`op06-completion-retrieval.log`](../../results/refinement-20261017/op06-completion-retrieval.log).
Final documentation checks also passed locally and on Linux (5 tests). The M1 live
runs exercised the real retrieval/LLM path. [Astra round 3](ASTRA_REVIEW_ROUND3.md)
found no blocking code defect; its review predates the final measurements below,
which root checked.

Each HTTP run used all 2,309 dev/noisy/hostile inputs, eight clients and 20 separate
warmup requests. Source results: `results/refinement-20261017/baseline_cpu/`,
`refined_cpu_v2/` and `refined_hybrid/`.

| Final measurement | Original CPU, Linux | Refined CPU, same Linux | Refined hybrid, M1 |
|---|---:|---:|---:|
| Intent accuracy | .7091 | .7091 | .7091 |
| Intent macro-F1 | .7015 | .7015 | .7015 |
| Action accuracy | .8117 | .8117 | .8361 |
| Clean / noisy action accuracy | .8145 / .7933 | .8145 / .7933 | .8385 / .8200 |
| Clean-to-noisy gap | .0212 | .0212 | .0185 |
| Top-70 error ratio | .3760 | .3760 | .4356 |
| Schema valid | 100% | 100% | 100% |
| Review share, labelled rows | 28.22% | 28.39% | 24.26% |
| Forbidden automatic suggestions, alias applied | 25 | 25 | 28 |
| Of those, high-risk | 3 | 3 | 3 |
| HTTP p95 | 134.946 ms | 136.184 ms | 1,504.782 ms |

All official development bars pass. The new CPU guard adds four public-dev reviews
without catching another forbidden suggestion: the confirmation effect size did
not transfer to these rows. The CPU's action confidence ranking is restored after
the cap fix. Compare CPU latency only within the matched Linux columns; the
hybrid uses a different machine and an external provider. These are single runs,
not a latency confidence interval or proof of hidden-test performance.

The final hybrid makes 460 measured LLM calls (463 including warmup); 421 successful
and 42 abstaining responses include warmup. Its measured cost is $0.093764 cached
list / $0.291252 uncached list for 2,309 inputs, about $0.041 / $0.126 per 1,000.
The intermediate capped hybrid scored .8330 action, gap .0303 and p95 1.5565 s;
the final run has 56 different action labels. Since the code change only affects
post-selection confidence, those label differences are not an action-accuracy
gain from uncapping: provider defaults and randomized request markers remain
variable. Both complete runs are retained; the final one is used because it
measures the corrected code, not because it has better noisy accuracy.

## Submission packaging and Linux CLI check

The canonical submission now points to `dev-http-llm-20261017`, the final hybrid
run above under policy `2026-10-17.2`. The top-level manifest, combined and split
prediction files, and latency file are the final-run package. The previous six
canonical files are preserved in [`results/history/2026-10-14/`](../../results/history/2026-10-14/);
all original raw run directories, including `results/raw/dev-http-llm/`, remain
unchanged. The manifest lists the 14 October run as historical and the 17 October
run as the current claim.

After promotion, the normal Linux CLI path was exercised with
`scripts/reproduce.sh http-eval completion-http-cpu-20261017` and LLMs disabled.
It returned all 2,309 dev/noisy/hostile responses with zero errors, non-200s,
schema failures, ID mismatches or replays; action accuracy was .8117, the other
CPU scores matched the table above, and Linux client p95 was 138.238 ms at eight
clients. The diagnostic output is retained in
[`results/raw/completion-http-cpu-20261017/`](../../results/raw/completion-http-cpu-20261017/),
with the CLI log at
[`op06-completion-cli.log`](../../results/refinement-20261017/op06-completion-cli.log).
The diagnostic preserved the canonical hybrid prediction/split/latency bytes and
the manifest's claimed run and metrics; the manifest also registers this CPU
diagnostic run.

## Expense and remaining limits

The new key's gateway meter reports **$0.19786692 spent, $9.80213308 remaining**.
This matches the recorded costs including both Claude attempts, both full Luna
runs, their warmups, and the three-call smoke. The uncached list-price total is
$0.5961055. These are separate cost measurements, detailed in
`results/refinement-20261017/spend.json`; they exclude older 14 October work and
assistant subscription/compute costs. An exact-value scan of 840 repository
files found no occurrence of the new credential; the closing scan also found no
match. The temporary credential file was outside the repository and has been removed.

The new work was agent-executed: Codex implementation/analysis, three Astra reviews,
and a Claude Haiku API critique checked against code. This is not an independent
human audit. Hybrid confidence still uses an unfitted agreement heuristic; true
flow/permission can be wrong despite high scores; small safety counts and the
public dataset limit transfer; and three high-risk forbidden automatic suggestions
remain in current public-dev evaluation. These are reasons to monitor outcomes
and review risk, not claims of general production safety.

## Reproduction

```
python experiments/refinement/run_state_blend.py prepare
python experiments/refinement/run_state_blend.py select
python experiments/refinement/run_state_blend.py confirm
python experiments/refinement/probe_gate.py
python experiments/refinement/evaluate_service.py --mode cpu --out /tmp/refined-cpu
```

The fitting/selection/confirmation commands write their own experiment outputs;
rerunning them is a new run, not new independent confirmation. API helpers accept
only an external `--key-file`; no credential is stored in this repository.
