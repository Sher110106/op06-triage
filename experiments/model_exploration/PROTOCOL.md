# Stronger model experiment, 17 October 2026

Model family and hardware remain open. The existing CPU plus Luna service is the deployed benchmark,
not a requirement to retain linear classifiers. This experiment tests a larger bidirectional encoder
with explicit observed state before any production switch.

## Candidate and hypothesis

The previous neural reference was a 32M-parameter Ettin encoder with a 256-token cap and an action head
that only saw mean-pooled text. The new candidate is `answerdotai/ModernBERT-base`: a 149M-parameter
encoder with a native long-context architecture. Use a 512-token cap for this bounded run, its exact
Hugging Face revision, and a shared encoder with intent, flow and action heads. The action head also
receives predicted intent probabilities and a learned projection of `T.state_dense` from observed
turns. Gold intent and future actions must never be inference features.
The action head's internal intent softmax stays at temperature 1, as trained; exported head temperatures
are applied afterward and do not change its input.

This changes architecture, size, input cap and state conditioning together. It tests this candidate
as a system; a positive result would not isolate which component caused the gain.

Primary sources checked 17 October:

- [Model card](https://huggingface.co/answerdotai/ModernBERT-base): size, architecture and Apache 2.0 licence.
- [Transformers documentation](https://huggingface.co/docs/transformers/model_doc/modernbert): supported loading and attention implementations.

## Data boundaries

Use only the published 20,000 training rows for fitting and decisions. `model_common.py` makes
conversation-disjoint fit/selection/calibration/confirmation partitions, approximately 65/10/10/15%,
with seeds 20261020, 20261021 and 20261022. Persist row IDs and verify disjointness and total counts.
This is a new re-split of previously studied data, not a historically pristine dataset or hidden test.

- Fit: neural weights, structured-feature scaling and a freshly fitted classical reference.
- Selection: checkpoint and ensemble-weight choice only. Criterion for the checkpoint is action
  accuracy plus .25 times intent accuracy. Max three epochs; learning rate 3e-5, effective batch 16,
  seed 20261020. A runtime benchmark may change batch size/accumulation or reject an infeasible run,
  without consulting confirmation outcomes.
- Calibration: per-head temperatures, correctness models and review cutoffs. The classical recipe
  receives these rows as its calibration fold. Grouped out-of-fold correctness scores select review
  cutoffs; shared temperature decisions still make calibration estimates optimistic.
- Confirmation: one frozen matched comparison after all choices. Do not retune using these outcomes.

Public dev/noisy/hostile are final regression diagnostics after freezing; do not select candidates or
blend weights on their scores. The existing shipped service remains a separate historical comparator,
because its training/calibration boundaries differ from this experiment.

## Comparisons and decision

Compare the refitted classical reference, encoder alone, and a probability ensemble. Select the
classical weight from {0, .25, .5, .75, 1} on selection, independently for intent and action; mix raw
action probabilities and flow probabilities before applying one policy prior. The policy version
and aliases remain fixed. Use one shared raw weight for action and flow so the action ensemble is
defined without another search dimension.
Ties favor more classical weight. The selection-frozen ensemble is the sole primary neural contrast;
encoder-alone results are secondary diagnostics. Selecting weight 1 for action is a valid negative result.

Both models use `gate.clean(project(row))`. The shared replay uses the actual `Triage._front/_back`
policy prior, observed-action masks, gate abstentions, permission guard and final review path. The current
default disables secondary-intent routing; the cached adapter does not reproduce its optional inference.
Intent-to-flow mappings come only from fit rows. This is an offline probability replay, not an HTTP
latency measurement or evidence that the encoder is already installed in the serving process.

Report paired conversation-bootstrap intervals for action and intent accuracy, macro-F1, error
concentration, and human-review outcomes under the stated 1/10/3 review/high-risk/low-risk cost model.
Report confidently forbidden suggestions against labelled flows, with the membership alias applied;
model-derived permission is not actual authorization.

A candidate merits production integration only if confirmation action accuracy improves by at least
one percentage point with a positive lower 95% interval endpoint, and intent accuracy loses no more
than one point. An offline pass is still conditional on confidence/routing validation, all published
schema/noise/top-70 bars, latency, memory, and actual service integration tests. Otherwise retain the
current default and publish the negative or inconclusive result.
The intent limit is a point-estimate guard, not a statistical noninferiority claim. Calibration scores
are partially OOF only: upstream temperatures use all calibration labels, as does classical calibration.
Before confirmation, also fix operational point guards for a default switch: no more than two extra
action-cost units per 100 cases, no more than five percentage points extra review, and no increase in
the high-risk forbidden-automatic rate versus the matched reference. Report intervals for these
quantities; passing point guards does not establish statistical noninferiority or permission safety.

## Resources and hosted models

Run one heavy compute process at a time. Bounded sequential API requests on the laptop may proceed
alongside remote training; they do not compete for that CPU or GPU, and neither experiment's model
choices may use the other's confirmation outcomes. The shared server's A6000 is occupied by another user's vLLM
process; do not interrupt it. Tyrone aliases were unreachable. Benchmark CPU feasibility first; use
GPU only when genuinely available. Record actual hardware/runtime, do not describe CPU training as
a GPU result. No paid GPU rental is authorized by this protocol.

Before training, eight-step CPU checks measured 3.678 / 2.063 / 1.863 seconds per optimizer step
with 4 / 16 / 32 threads. Select 32 threads for the full run. Before examining selection outcomes,
raise the runner's default 45-minute time cap to 90 minutes so the declared three epochs can finish
on this CPU. Model weights, learning rate, split and selection criterion remain as specified above.
Fit contexts average 94.9 tokens (p95 137, maximum 278), so the 512-token cap does not truncate fit;
this test does not assume a larger context window itself improves accuracy.

### Bounded head refinement before freezing

Before final base-training results are available, add one targeted training refinement: the fresh
action/state MLP used the encoder's small learning rate. After base training, freeze encoder, intent
and flow weights; extract fitting/selection pooled representations and train only the action/state
branch at learning rate .001 for at most 20 epochs, batch 256, seed 20261025. Include the unchanged
base head as epoch zero, select the head checkpoint with the same selection criterion, and use the
refined neural model only if raw selection action accuracy improves by at least .005. Otherwise use
the base model. Then perform temperature/calibrator/ensemble selection once for that neural choice,
and one primary confirmation comparison. Neither calibration nor confirmation labels enter head
training or head selection. This is selection-guided model development, not an untouched selection
estimate; the held-out current-round confirmation remains the check of the whole chosen recipe.

The previously authorized OpenRouter budget has approximately $9.80 remaining. Any additional hosted
model comparison must first freeze its task, model ID, retrieval corpus, sample and conservative spend
cap, then meter every attempt. Credentials stay outside the repository and outputs. No hosted-model
gain is claimed until its actual predictions are scored.

### Bounded hosted comparison

Secondary exploration compares the current Luna family with `anthropic/claude-haiku-5.5` and
`google/gemini-3.8-flash` on the same fit-only retrieved examples and unchanged V3 prompt. Freeze a
seed-20261023 random sample of up to 300 eligible low-reference-confidence selection rows; do not
sample by gold errors. Choose the model by full-cohort action accuracy with failures retained as CPU
fallbacks, breaking ties by lower measured cost. Do not use model-stated confidence for selection.
If accuracy-tied candidates do not all have complete reported costs, label the cost tie unresolved
and retain Luna when it is among the tied candidates; otherwise use run ID only as a deterministic
fallback. Unknown billing is not evidence that a model costs more. Fix this missing-data rule
before examining selection scores.
Eligibility uses the uncapped CPU confidence before final policy-review caps, from the actual Triage
trace, matching the service's LLM gate. Frozen fallbacks preserve the final response separately.
Then freeze the winner and compare it with the same current Luna control on the eligible confirmation
tail, using a seed-fixed random cohort capped at 300 rows for the shared $5 budget. Fix this cap before
selection scores are examined. If the control wins selection, retain it without paying for a redundant
self-comparison. Report tail accuracy and its paired interval, plus the observed all-row effect with
unqueried rows unchanged; this is not a claim about applying the model to the entire unqueried tail.
neural and hosted contrasts are separate. This hosted result is secondary, not another opportunity
to choose a primary confirmation winner after inspecting neural results.

Record every attempt, returned provider/model, token usage and actual gateway expense. The phase's
hard new-spend ceiling is $5, inside the previously authorized remaining budget; use conservative
pre-call reservations and bounded per-run limits. Price sources are the saved live gateway catalog.
Use 30-second offline deadlines to measure model quality separately from the production 4-second
deadline. Luna and Haiku use temperature 0, reasoning disabled and 256 maximum output tokens. Three
Flash compatibility calls rejected reasoning disabled; preserve those failures separately. Google's
[current documentation](https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.8) says 3.8 Flash
requires low/medium/high thinking. Its supported test profile uses low reasoning, native sampling
defaults and 1,024 maximum output tokens. This compares model configurations, not model weights alone.
Fixed cohort prompts also differ from historical provider-default sampling.
Any deployment change still requires the actual production deadline and service regression checks.
