# Stronger-model comparison, 17–18 October 2026

The frozen classical/neural ensemble improved confirmation action accuracy from 80.59% to
84.26%, but failed the predeclared operating guard: high-risk automatic suggestions forbidden
under the labelled-flow metric increased from two to five. The existing default is retained.

This extends the completed 17 October refinement. The existing classical-plus-Luna service is
the historical deployment comparator. Model family and hardware were open for this experiment;
CPU training was an available execution resource, not a requirement to keep linear models.

## What we tested and why

The earlier 32M Ettin experiment retained probability arrays but no usable checkpoint and did
not establish deployment-ready confidence. We trained a new `answerdotai/ModernBERT-base`
candidate, pinned to revision `8949b909ec900327062f0ebf497f51aef5e6f0c8`. Its shared encoder
feeds intent, flow and action heads. The action branch also receives its own predicted intent
distribution and 138 observed-state features projected to 64 dimensions. Gold intent and the
future action are never features. The architecture changes size, encoder and state conditioning
together, so the comparison cannot attribute a gain to model size alone.

The practical hypothesis is that contextual text features complement the strong sparse/state
reference. It is not a long-context hypothesis: fitting inputs averaged 94.9 tokens, p95 137,
maximum 278, all below the 512-token cap. CPU feasibility checks selected 32 threads after
4/16/32-thread checks measured 3.678/2.063/1.863 seconds per training step. The shared GPU was
occupied; it was left alone. CUDA remains supported by the scripts but unvalidated in this run.

The independent hosted branch tested Luna, Haiku and Gemini Flash as action-only advisers on
the same frozen low-confidence cases and eight fit-only retrieved examples. It does not test
a neural-plus-LLM combination. See [ARCHITECTURE.md](ARCHITECTURE.md) for a presentation guide,
[PRIOR_MODELS.md](PRIOR_MODELS.md) for the earlier experiments, and [README.md](README.md) for commands.

## Evaluation boundaries

The 20,000 published training rows were split by conversation into 12,999 fit, 2,014 selection,
1,957 calibration and 3,030 confirmation rows. The corresponding conversation counts are
5,044/777/777/1,165. These are new current-round boundaries on previously studied public data,
not a historically pristine dataset or a hidden test.

Both model families use the same cleaned observed input. The reference was refitted on the
same fitting groups. Selection chooses checkpoints and blend weights; calibration fits
temperatures, correctness predictors and cost-based review thresholds. The correctness
out-of-fold scores exclude conversations only in the final regression; upstream temperature
and classical calibration decisions share calibration labels. Treat calibration estimates
as optimistic. Frozen confirmation is the primary comparison; repeatedly used public dev,
noisy and hostile sets are regression diagnostics.

The shared replay uses the actual gate, policy prior, mask support, permission check and review
path. The current policy has no active prerequisite-mask rules. A flow-derived permission
score is predictive support, not actual authorization. Both accuracy and automatic forbidden
suggestions are reported. Review/action error costs of 1/10/3 are modeling assumptions, not money.

The neural integration rule requires at least one percentage point of confirmation action
gain with a positive lower 95% paired interval, no more than one point of intent loss, and the
operational guards fixed in [PROTOCOL.md](PROTOCOL.md). These point guards do not establish
statistical noninferiority or permission safety. Experiment source and rules were committed
as `a53e64c` before confirmation scoring, while training was still running.

## Neural training and the targeted refinement

The base run completed all 2,439 optimizer steps over three epochs. Its reported elapsed time
was 76.92 minutes with 32 CPU threads on two Xeon Gold 6330 sockets. The selection-best checkpoint
at step 2,436 had raw action/intent/flow accuracies of 81.73%/72.24%/86.79%; the final three steps
did not change those accuracies. These raw selection scores neither include the policy prior
nor estimate generalization after model choice.

The preplanned refinement freezes encoder, intent and flow weights and trains only the action/
state branch at learning rate .001, versus the base joint run's .00003. It includes the unchanged
base head as epoch zero and requires at least .005 raw selection action gain to use a refined
checkpoint. It addresses possible under-training of a newly initialized branch without claiming
that a larger model must win. Cached pre-dropout features are checked against direct forward
logits. The synthetic real-Torch smoke verified this seam, frozen weights and checkpoint reload.
The 20-epoch refinement reached its best selection action accuracy, 81.93%, at epoch 17:
four additional correct rows out of 2,014, or +0.20 percentage points. That missed the fixed
0.50-point minimum, so the base checkpoint was retained. Falling training loss did not justify
using the more aggressively fitted head.

## Frozen neural confirmation and deployment decision

Selection chose equal classical/neural weights for intent, action and flow. Neural temperatures
were action .936940, intent .921048 and flow .994982. Calibration fitted separate correctness
predictors for each candidate. Reference high/low review cutoffs were .897663/.719110; ensemble
cutoffs were .799597/.722496. These are different fitted confidence scales.

The primary comparison used the same 3,030 confirmation rows in 1,165 conversations:

| Metric | Matched classical reference | Encoder alone, secondary | Frozen 50/50 ensemble |
|---|---:|---:|---:|
| Action accuracy | 80.59% | 82.81% | 84.26% |
| Intent accuracy | 71.25% | 71.49% | 73.83% |
| Intent macro-F1 | .7027 | .7001 | .7285 |
| Top-70% error ratio | .3885 | .3508 | .3564 |
| Human reviews | 853 (28.15%) | 797 (26.30%) | 699 (23.07%) |
| Assumed action/review cost per 100 | 47.00 | 45.51 | 43.20 |
| Automatic forbidden suggestions | 34 | 45 | 34 |
| High-risk subset of those suggestions | 2 | 6 | 5 |

The ensemble's exact action gain was +3.6634 percentage points, paired conversation-bootstrap
95% interval [+2.7610, +4.5857], using 5,000 resamples. Intent gained +2.5743 points
[+1.5516, +3.5998]. Review rate fell by 5.0825 points [−6.3164, −3.8590], and modeled cost
fell by 3.7954 units per 100 [−6.4779, −1.0134]. The action and intent accuracy criteria passed.

The operating guard failed because the high-risk forbidden-automatic count rose by three rows
in two conversations. Its rate difference was +0.09901 percentage points [0, +0.26552]. This
is a conservative point-rule failure, not a statistically established population increase in
unsafe behavior. No thresholds, permissions, blend weights or checkpoint were retuned after
seeing this outcome. The candidate is retained as a runnable experimental backend, not promoted.

Posthoc [case analysis](FAILURE_ANALYSIS.md) found that the three additions had the same action
and intent in both systems: they became automatic under the ensemble's new confidence/routing
combination. Every case exceeded the .90 predicted-permission support guard. All five high-risk
cases also have gold actions outside their recorded permission lists; three predictions match
those gold actions exactly. This is the task's action/policy annotation conflict in concrete form.
It limits the interpretation of “forbidden” and does not justify deleting cases or relaxing the
frozen guard. Action-correctness confidence is not proof of authorization; no business action ran.
The cost decomposition gives another caution: total wrong automatic actions fell from 188 to
173, but their high-risk subset rose from 1 to 13. The review savings outweigh those errors at
the assumed high-risk cost of 10. Posthoc valuation of the same frozen decisions loses that
advantage at a high-risk error cost of 19.5833 human reviews; at cost 20 the ensemble is .1650
units per 100 more costly. This sensitivity does not refit thresholds or change the frozen rule.

## Public regression and implementation checks

After freezing, the unchanged official grader scored joined dev/noisy/hostile predictions:

| Public metric | Matched reference | Encoder alone | Ensemble |
|---|---:|---:|---:|
| Action accuracy, 2,300 labelled rows | 80.96% | 81.00% | 82.83% |
| Intent accuracy | 69.78% | 70.74% | 72.09% |
| Intent macro-F1 | .6749 | .6753 | .7000 |
| Clean / noisy action accuracy | 81.30% / 78.67% | 81.80% / 75.67% | 83.30% / 79.67% |
| Clean-to-noisy gap | .0263 | .0613 | .0363 |
| Top-70% error ratio | .3981 | .3989 | .3326 |
| Schema validity, all 2,309 inputs | 100% | 100% | 100% |

All three clear the published development bars. These repeatedly used public rows are regression
diagnostics. The current 17 October hybrid's 83.61% action accuracy is a separate historical
public-data comparator with different training boundaries and an LLM step; it is not the matched
classical baseline in the confirmation table.

Real inference matched the frozen replay on 100 selection rows and ten singleton checks, with
the requested two Torch threads verified on the request worker. Live confidence uses the selected
ensemble's fitted predictor and cutoffs; the service freezes the matching policy version at load.
The public reporting path exposed two helper bugs, now covered by regressions: a noisy-only gap
was computed without a clean denominator, and hostile null targets were mistaken for labels.
Neither fix changes the recorded all-clean primary confirmation calculation or model decisions.
The complete Linux suite then passed **546 tests, with zero skips**, in 158.64 seconds. An earlier
isolated-copy run lacked canonical historical-result fixtures; the existing evidence files were
copied into that workspace before the complete rerun. Failed attempts and successful outputs
are retained under `validation/` and `feasibility/`. A final additive manifest index then passed
four focused metadata/provenance checks; the canonical claims and five prediction/latency
artifacts remained unchanged.

## Actual neural HTTP serving

Single-request predictor p95 on 100 selection rows, including the separate latest-customer
probe, was 396.11/234.14/168.41 ms at 1/2/4 Torch threads. These are predictor probes without
HTTP. Four threads were used for the full service test on the stated Xeon host.

The actual optional backend answered all 2,309 public inputs over HTTP at eight concurrent
clients, with 20 distinct-ID warmups. There were zero transport errors, non-200 responses,
invalid schemas, ID mismatches or cache replays. Client latency was p50 1,207.190 ms,
p95 1,366.997 ms, p99 1,498.926 ms and maximum 1,537.340 ms; throughput was 6.58 requests/second.
Peak server RSS, including startup, was 2,222,056 KiB (2.12 GiB); resident RSS after the load
was 1,631,952 KiB (1.56 GiB). The LLM step made zero calls and incurred $0 model-API cost;
hardware, electricity and hosting are not priced by that counter.

All served intent/action labels and review flags matched the frozen offline outputs. Maximum
confidence difference was .000003, within the .00001 tolerance. The official grader also scored
the HTTP responses. The server was shut down after measurement. This makes the alternative
runnable and measured; it does not reverse its failed operating guard. CPU, neural and historical
M1-hybrid latencies use different processes or hardware and are not a controlled speedup claim.

## Hosted selection: a quality and latency tradeoff

All three models attempted the same 300 randomly sampled cases from 393 eligible low-reference-
confidence selection rows. CPU fallback accuracy on those 300 was 45.33%. Failures, invalid
outputs and abstentions stay in the denominator and retain that fallback.

| Requested model | Tail action accuracy | Four-second action replay | Provider-call p95 | Reported cost, 300 calls | Uncached estimate |
|---|---:|---:|---:|---:|---:|
| `openai/gpt-6-luna` | 50.00% | 50.00% | 2.169 s | $0.06246 | $0.19065 |
| `anthropic/claude-haiku-5.5` | 51.33% | 51.33% | 1.822 s | $0.30662 | $0.30662 |
| `google/gemini-3.8-flash` | 59.33% | 55.33% | 18.126 s | $1.60567 | $1.76624 |

The frozen selection rule chose Flash. Its supported profile uses low reasoning, native sampling
and a 1,024-token output cap; Luna and Haiku use temperature zero, reasoning disabled and 256
output tokens. This compares configurations, not only model weights. The earlier unsupported
Flash profile and controlled HTTP 400 diagnostic are preserved separately and are not pooled
with the supported run. Flash accepted 279 actions, abstained 19 times and returned two invalid
JSON outputs; Luna accepted 271 and abstained 29. Only 72.67% of Flash calls finished within four
seconds, compared with 100% for the other two models.

The four-second column retrospectively keeps the CPU answer when a recorded call exceeds four
seconds. It is not a measured HTTP-service benchmark, a cancellation test, or an estimate of
what the provider would bill after cancellation. All-row figures change only the 300 queried
rows and leave 1,714 selection rows unchanged; they do not claim an effect from querying the
entire eligible tail. Selection results choose the hosted challenger, not a deployment winner.
Flash versus Luna on selection was +9.33 percentage points, paired conversation-bootstrap
95% interval [+3.19, +15.38]. Under the four-second replay it was +5.33 points [−0.65, +11.15].
Those intervals describe the development sample used to choose the challenger. Hosted IDs are
provider aliases; recording requested and returned IDs does not prove an immutable weight snapshot.

## Hosted confirmation and provider decision

The frozen challenger, Flash, and current Luna control each attempted the same 300 confirmation
cases, randomly selected from 541 eligible rows. The sample spans 252 conversations. The CPU
fallback was correct on 44.00% of this difficult tail. Every failure, abstention and invalid
output retained that fallback.

| Confirmation-tail metric | Luna | Flash, low reasoning |
|---|---:|---:|
| Action accuracy, 30-second call deadline | 61.33% | 67.00% |
| Action accuracy, four-second retrospective fallback | 61.33% | 66.00% |
| Provider-call p95 | 1.985 s | 5.187 s |
| Calls finishing within four seconds | 299/300 | 262/300 |
| Accepted actions | 275 | 288 |
| Abstentions / invalid outputs | 25 / 0 | 10 / 2 |
| Reported charge, 300 calls | $0.06275 | $1.62356 |
| Uncached list estimate, 300 calls | $0.19088 | $1.78592 |

Flash's 30-second tail gain over Luna was +5.67 percentage points, paired conversation-bootstrap
95% interval [+1.01, +10.56]. Under the four-second replay the gain was +4.67 points [0, +9.67],
not strictly positive at the lower endpoint. These intervals condition on the observed provider
responses and sampled conversations; they do not include another round of model sampling or
future provider changes. Latency changed substantially between selection and confirmation and
is not a stable SLA estimate.

Across all 3,030 confirmation rows, changing only the 300 queried rows produced 82.31% action
accuracy for Luna and 82.87% for Flash at 30 seconds. At four seconds Flash produced 82.77%:
an observed +0.4620-point gain over Luna, interval [−0.0327, +0.9709]. The other 2,730 rows stayed
unchanged. Do not extrapolate these numbers to querying all 541 eligible rows or to a neural
uncertain tail, which would contain different cases.

Flash demonstrates better action quality with a longer deadline. Its current-deadline advantage
remains uncertain and its uncached cost was about 9.4 times Luna's. This action-only experiment
does not calibrate a replacement's review confidence or measure an actual Flash-backed service
under concurrent load. Retain Luna in the default pipeline; do not claim a deployed provider upgrade.

## Budget closure and artifacts

The phase made 1,504 recorded attempts: 1,500 calls in five complete 300-row comparisons and
four preserved compatibility/error diagnostics. The gateway's settled additional charge was
**$3.66106068**, matching the sum of returned charges. The uncached list estimate was $4.24029855.
Conservative accounting was $4.395242575, including $0.1546425 of unknown-usage reservations.
All remain below the phase's $5 ceiling. The key's earlier usage was $0.19786692; total usage
is $3.85892760 and remaining allowance is $6.14107240 of the authorized $10. Cumulative reservation
totals in raw journals are not extra spending: completed requests release their reservation to
known liability. These costs exclude local hardware and coding-agent subscription costs.

The selected inference package contains the encoder checkpoint, tokenizer/config, fitted reference,
calibrators, frozen blend/cutoffs, hardware record and upstream attribution. It is 635,136,000 bytes;
SHA-256 `523dd618b63798a58c2a4ae4d4117e1298b80a0108d2f4d9ed96d312391d1af8`.
The checkpoint was also reloaded and exercised through the real HTTP backend. Checkpoints stay
outside Git; aggregate and raw experiment evidence stays in the private repository. The public
explainer receives aggregate findings only. No hiring form or reviewer invitation was submitted.

Raw evidence is under `results/model-exploration-20261017/`. The exact split IDs, frozen choice,
prompts, returned providers/models, successful and failed attempts, usage and reservations are
retained in the private repository. A final exact-credential scan found no match in 1,713
deliverable files, and the temporary external credential file was removed after budget closure. Public explainers
contain aggregate results rather than raw provider requests.
