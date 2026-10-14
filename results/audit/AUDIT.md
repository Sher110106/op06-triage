# Independent audit of the OP-06 submission

Audit date: 14 Oct 2026, working tree on top of commit `21f486d`. I read the files and ran the code. I changed no
shipped code. Everything I wrote is under `results/audit/`, plus the one run the task asked for,
`results/raw/audit-cpu/`. That run also added one entry to `results/manifest.json`, because `reproduce.sh eval`
rebuilds the manifest. The routing rerun rewrote `results/raw/dev-pipeline-llm-replay/traces.jsonl` (see check 6).

Scripts and outputs in this folder:

| File | What it does |
|---|---|
| `score.py` | My own OP-06 scorer, written from the problem statement (not `harness/eval.py`, not `grader.py`) |
| `headline.py`, `headline.json` | Headline numbers: my scorer against the official report and the manifest claim; latency and cost from the per-request records |
| `spend.py`, `spend.json` | Spend recomputed from the ledger's token counts at list price |
| `leak_test.py`, `leak_test.json` | Runs the pipeline on projected rows, full gold rows and rows with scrambled gold fields |
| `reproduce_eval_audit-cpu.log`, `analyse_routing_rerun.log` | Logs of the two reruns |
| `snapshot_before_routing/` | Copies of the routing outputs taken before the rerun |

## Summary

The submitted numbers are real. My own scorer gives the same values as the official report for
`results/predictions.jsonl`. The pipeline reads no gold field. Both LLM pre-registrations were pushed before
the first call. Spend is under 1/10 of the cap. Both reruns reproduce the committed outputs.

The package is not ready to submit:
- **The manifest claims a different system from the one whose predictions it ships** (critical).
- **`MEMO.md` is a placeholder** (critical).
- The log has no entry for the submitted run.
- The pre-registered encoder test is missing from the log.
- Several log and decision numbers don't match their cited files.
- Two of the six Linux test failures are not the ones the log describes.

## 1. Headline numbers, recomputed

`results/predictions.jsonl` is byte-identical to `results/raw/dev-http-llm/predictions.jsonl`. The three split
files (`dev_predictions.jsonl`, 2,000 rows; `noisy_predictions.jsonl`, 300; `hostile_results.jsonl`, 9), put
together, are exactly that file.

| Metric (dev 2,000 + dev_noisy 300; schema over all 2,309) | My scorer | `report_official.json` | `manifest.json` claimed |
|---|---|---|---|
| Intent accuracy | 0.7091 | 0.7091 | 0.7091 |
| Intent macro-F1 | 0.7015 | 0.7015 | 0.7015 |
| Action accuracy | 0.8365 | 0.8365 | **0.8117** |
| Action clean / noisy | 0.8395 / 0.8167 | 0.8395 / 0.8167 | noisy_accuracy **0.7933** |
| Noisy gap (clean − noisy) | 0.0228 | 0.0228 | (none) |
| Top-70 error ratio, ties kept (1,610 rows kept, cut 0.796) | 0.4331 | 0.433 | **0.376** |
| Schema valid | 2,309 / 2,309 = 100% | 100% | 100.0 |
| Policy violation (gold's own) | 0.0848 (0.0883) | 0.0848 (0.0883) | (none) |
| needs_human share (labelled rows) | 24.2% (clean 22.8%, noisy 33.7%) | | |

The 0.4331 against 0.433 gap comes from rounding: the grader rounds both accuracies to 4 places before it divides.

The manifest's `claimed` block comes from `dev-classical-v0` (`claimed_from_run`, set by `CLAIM_RUN` in
`harness/manifest.py`). That run is the CPU model with the old single threshold, not the submitted service. Three
claimed values differ from what a reviewer will compute from `results/predictions.jsonl`.

**Latency and cost**, from the 2,309 per-request records in `results/latency_bench.json`. The run was over HTTP
on an M1 laptop at concurrency 8, with the real Luna step on.

| | Recomputed | File says |
|---|---|---|
| p50 | 8.9 ms | 8.877 |
| p95 | 1,656.5 ms (linear), 1,656.8 ms (nearest rank) | 1,656.471 |
| p99 | 2,432.5 ms | 2,432.497 |
| Status 200 / schema valid / id echoed | 2,309 / 2,309 / 2,309 | |
| Requests over 500 ms | 460 (= the 460 LLM calls) | |
| Cost per message | $0.0000404 ($0.040 per 1,000), list price with caching | $0.0000404 |

The per-request records carry no cost. The cost is the service's own `/metrics` delta, so I could only redo the
division, not the measurement. At the fold-B uncached price per call ($0.000632), the same 460 calls would
cost about $0.126 per 1,000 messages. The manifest still has `cost_per_message_usd` and `latency_p95_s` as
null, with the note "stay null until the Phase 3 load test".

## 2. Numbers in EXPERIMENT_LOG.md and DECISIONS.md against their files

### Checked and matching

- **P0-2:** baseline 0.3748 / 0.4508 / 0.3913 / 0.0743 / 1.0 / 0.1404. My scorer gives the same.
- **P0-3:** every number in the table, including the intervals, routing 34.7%, unrouted accuracy 0.936, coverage 0.955 / 0.929 / 0.862, and the per-noise-kind table. Source: `dev-classical-v0`, my scorer and `report.json`.
- **P0-4:** the systems table and its intervals, the exploratory share table, the costs ($0.47 / $1.46; $0.72 / $2.48), and the p50 / p95. Sources: `experiments/llm_tail/results.json` and the ledger.
- **P0-5:**
  - fold-B CPU 0.745 / 0.833, gate share 16.7% and 51.9%, the action step +0.0247 [+0.017, +0.032], top-70 0.324 → 0.380, the intent steps, costs and latency;
  - rule selection 0.750 / 0.747 / 0.730 / 0.712.
  - Sources: `results_B.json` and `rule_selection_dev.json`.
- **P1-1:**
  - `dev-http-cpu`: p50 26 ms, p95 38 ms, 302 req/s;
  - zero injection or junk flags over train, dev and dev_noisy (I re-ran the gate on all 22,300 rows);
  - 464 tests at that commit (the Linux log shows 458 passed and 6 failed).
- **P2-1:** the per-tier rows, the off-policy cost (+0.55), off-policy accuracy 63% / 57%, second intent 87% and 19 of 50 at 95%, and the dev result table (0.8361 [0.815, 0.857], gap 0.034, top-70 0.439, violation 0.0857 / 0.0465, 24.0%, 0.914). LLM off: 28.2% routed, 0.920.
- **P3-1:** 0 intents, 5 actions and 4 needs_human differ; largest confidence difference 0.036; 0.8104 against 0.8117.
- **P3-3:** p50 / p95 / p99 and throughput at concurrency 8 and 80, recomputed from the records.
- **D2 tables:** these match the same files.

### Mismatches

| File:line | Says | Cited file says |
|---|---|---|
| EXPERIMENT_LOG.md:245 | v0 threshold on fold B: routed 30.1%, auto-handled 0.946, cost 44.0 | `experiments/routing/results_routing.json` `fold_B.v0_threshold_0.8498`: 31.06%, 0.9466, 44.58 |
| EXPERIMENT_LOG.md:260 | At 5 / 3, fold B routes 14.8% | `sweep_fold_B["R_high=5,R_low=3"]`: 16.41% |
| EXPERIMENT_LOG.md:261 | At 20 / 10, fold B routes 39.5% | `sweep_fold_B["R_high=20,R_low=10"]`: 40.17% |
| EXPERIMENT_LOG.md:267 | "6 parity tests failed", all put down to the solver | `linux-retrained-cpu/reproduce_all_linux.log`: two of the six are something else. `test_retrieval_matches_llm_test_examples` matched 198/200: the retrieval index built on Linux differs. `test_eight_big_turns_are_fast` took 0.545 s against a 0.5 s limit. |
| EXPERIMENT_LOG.md:230 | Against v0, "confidence and routing differ only on rows a flag routes" | `dev-pipeline-cpu` was regenerated at 16:33 with the Phase 2 thresholds. needs_human differs from `dev-classical-v0` on 264 of 2,300 rows. Confidence differs on only 51 rows, all capped at 0.5, as stated. |
| EXPERIMENT_LOG.md:233 | `dev-http-cpu`: "every body equals the in-process answer" | True against `dev-pipeline-cpu-e2e` (policy .1). Against the current `dev-pipeline-cpu`, needs_human differs on 237 rows. |
| EXPERIMENT_LOG.md:231 | `no_dialogue` fires on 1 dev row | It fires on 4432-4, on its noisy twin, and on 12 train rows. |
| EXPERIMENT_LOG.md:67 (R6) | De-dup plus the 2,000-character cap moved very_long intent from 0.10 to 0.68 (cites `exp_n0_raw.json` against `exp_n5_dedup600.json`) | The "after" file uses a 600-character cap plus NFKC, and the "before" file has no cap. The like-for-like pair, `e6_noise_final.json` (no de-dup against final), is 0.16 → 0.68. |
| DECISIONS.md:13 | Hard mask: "violations are near zero" | Probe `report_a_direct…__hard_alias__stacked_LR.json`: 0.0774. Repo `ablation-hard-mask` (exploratory, dev + noisy): 0.0743. The grader scores violations against the gold flow, so masking by the predicted flow cannot reach zero. |
| DECISIONS.md:57–62 | Per-flow gains (+11.5, −11.4, …), +0.047 over "40 intents", and "noisy 1.3 points worse", placed under the line that says the primary analysis uses 1,940 fresh rows | These are all-2,300-row figures (my recount: +11.6, −11.5, +0.047 over 40 intents, noisy −1.3). On the 1,940 fresh rows: manage_account +10.4, shipping −9.5, +0.046 over 48 intents, noisy −3.9. No file in the repo stores this analysis. |
| DECISIONS.md:75 | Fold B: rows "which no model choice had seen" | Fold B chose the confidence stacker, the temperatures and the 0.8498 threshold (R2, R5), and later the P2 routing thresholds. The fold-B pre-registration says so itself. |
| EXPERIMENT_LOG.md:163, 208 | Cumulative OpenRouter spend $1.75 and $2.55 | These are key-balance figures (keycheck 1.7485 and 2.5457). The ledger plus the probe gives $1.76 and $2.56. Not wrong, but the source is not named. |
| EXPERIMENT_LOG.md:282–295 (P3-2) | Thread tests: 105–136, 40–60, 127 and 137 req/s | No raw file is cited or present. Unverifiable. |

### Reconstructed R-entries

R1 to R10 can be checked against the 13 Oct files in `/Users/sher/.t3/projects/work-01/scratch/`.

**Checked and matching:**
- **R1:** 1,228 intent errors (518 / 408 / 235); 1,198 action errors (460 / 283 / 225); 94 off-policy plus 50 alias. Source: `probe_classical/out_remote/e1_baseline_taxonomy.json`.
- **R4:** action 0.8078 (≈ 0.808); wrong flow 114 → 64; state ignored 10 → 81; violations 0.0804 / 0.0448, gold 0.0883 / 0.057. Sources: `e1_compare.json` and `final_grade.json`.
- **R5:** 2,280 distinct confidences.
- **R6:** the action gap 0.0502 → 0.0235.
- **R7:** 0.7091 / 0.7015 / 0.8104 / 0.0235 / 0.3734; p50 8.02 ms, p95 14.62 ms; 72 MB / 35 MB.
- **R8:** Ettin 0.733 / 0.716 (clean dev) / 0.834 / 0.793; 25.1 minutes. Source: `probe_encoders/logs/ft_ett32.json`.
- **R9:** Luna −0.022 [−0.064, +0.017] (`probe_llm/report_tables.md`). Spend: Anthropic $0.4318, OpenRouter $1.2858 reported (`probe_llm/ledger.jsonl`). The ledger also shows 5 OpenRouter HTTP errors that R9 does not mention.
- **R10:** hostile-002, 003 and 006 confident and unrouted; 007 and 008 abstain (`e9_latency_t2.json`).

**Not checked or unverifiable:**
- R2: the view ablations (0.515, 0.226, 0.628 / 0.856); I did not find these figures in `e2_views.log`.
- R3: the parser rules.
- The R8 frozen-embedding figures.
- The R9 Artificial Analysis correlation.

Also, the R8 / encoder pre-registration motive compares encoder *clean-dev* action (0.834) with classical
*pooled* action (0.810). The probe's classical clean figure is 0.8135.

## 3. Leakage

- **Request path.** `Triage._gate` rebuilds each row as `{id, turn_index, context[speaker, text]}` before the CPU or LLM stage sees it.
  - I ran the pipeline on all 2,309 dev, noisy and hostile rows three ways: projected, full gold rows, and gold fields set to wrong values. The 2,309 responses were identical across all three.
  - I repeated this with a stub LLM step on (460 calls per pass). The only keys that reached the stub were `context`, `id` and `turn_index` (`leak_test.json`).
- **The only gold reads in `triage/`:**
  - training code (`classical.py` fit);
  - the intent → flow map read from train (`pipeline.intent_flows`);
  - the retrieval payloads built from train.
- **Policy file.** All 10 flow lists in `policy/policy.toml` equal the lists in train, and dev's lists are the same. `scripts/build_policy.py` reads train only.
- **Retrieval.**
  - The dev test's examples (`experiments/llm_tail/retrieval.json`, 7,568 ids) are all train ids, and none is a dev id.
  - The fold-B index (`experiments/llm_confirm/retrieval_B.json`) covers exactly the 3,966 fold-B rows. All 9,080 example ids are in fold A, and none shares a conversation with fold B.
  - The fold A / B split shares no conversation.
  - The LLM runners pass `strip_gold(row)` to the prompt.
- **Caveat.** The 0.642841 gate and the 13 Oct configuration were chosen on dev. So the dev headline (0.8365) sits on rows that helped choose the system. Fold B (+0.025) is the clean estimate of the LLM step's gain.

## 4. Pre-registration timing

Commit and push times come from `git log` and the `origin/main` reflog. Ledger timestamps are local time
(+05:30): the ledger file's mtime, 13:59:52+0530, equals its last record.

| Test | Pre-registration commit | Pushed | First API call (ledger ts − latency) | Margin |
|---|---|---|---|---|
| Full-dev LLM test | `7f4edfb` 13:19:47 | 13:19:57 | 13:20:05 (luna smoke) | 8 s after push |
| Fold-B confirmation | `bd6fc45` 13:44:58 | 13:45:01 | 13:45:02 (luna_B smoke) | about 1 s after push |
| Encoder (no API) | `d1f7258` 14:36:34 | 14:36:36 | outputs written 14:45:26–28 (file mtimes) | outputs after commit |

None of the three PREREGISTRATION.md files changed after its commit.

The encoder pre-registration says it was written while training ran. A 10.9-minute run that ended about 14:45
started about 14:34, before the commit. Fold-B and dev outputs only exist at the end, but this cannot be proved
from the files, because they hold no internal timestamp. The text discloses this.

Two integrity points a reviewer will raise:

1. **The first pre-registration's rule passed system B, and D2 overrode it.** B passed on macro-F1: +0.031 [+0.004, +0.054], and `results.json` records `adopt_B: true`. D2 explains the override.
2. **The fold-B pre-registration changed the intent bar to accuracy only.** Under the first rule's macro-F1 criterion, both fold-B intent steps pass: Luna +0.0202 [+0.0072, +0.0323], Haiku +0.0219 [+0.0081, +0.0371]. D2's "under the bar set in advance" is true for the second rule, but D2 does not name the change.

## 5. Spend

My arithmetic uses the token counts in each ledger line. It matches the ledger's own cost on all 10,646 lines.

| Source | Provider | Calls | List price with caching | List price uncached |
|---|---|---|---|---|
| Repo ledger, 14 Oct (`results/raw/llm-ledger/ledger.jsonl`) | OpenRouter (Luna) | 6,276, all ok | $1.2774 (equal to billed) | $3.9720 |
| | Anthropic (Haiku) | 4,370, all ok | $1.4068 | $4.7389 |
| `dev-http-llm` service run (from the service's `/metrics`; **not in the ledger**) | OpenRouter (Luna) | 463 (460 + 3 warm-up; 3 timeouts, 36 abstentions) | $0.0944 | about $0.29 (estimate) |
| 13 Oct probe (`scratch/probe_llm/ledger.jsonl`, outside the repo) | Anthropic | 1,412 | $0.4318 | $1.4845 |
| | OpenRouter (6 models) | 2,102 (5 HTTP errors) | $1.2858 billed ($1.1457 list) | $1.8445 |
| **Total** | | | **about $4.50** | **about $12.33** |

Total spend is under the US$50 ceiling by a wide margin. Runs in `results/` alone come to $2.78 with caching.

Model snapshots used in `results/`:
- `openai/gpt-6-luna-20260922` requested, OpenAI host pinned. Every response returned the model id `openai/gpt-6-luna`.
- `claude-haiku-5-5`, returned as the same id.
- `jhu-clsp/ettin-encoder-32m`, Hugging Face revision not recorded.
- CPU bundles: `c007c7a3…` (shipped), `98f2da30…` (Linux retrain), and `17bbb0ab…` and `5596a6da…` (ablations).
- The published baseline.

The probe also used Qwen3.8 27B, Gemma 4 31B, Gemini 3.5 Flash-Lite, Gemini 3.8 Flash and Solar Mini 4 by
alias, outside `results/`.

## 6. Reproduction

- **`TRIAGE_LLM=off scripts/reproduce.sh eval audit-cpu`** (exit 0, 3.0 s batch).
  - `predictions.jsonl` and `report_official.json` are byte-identical to `dev-pipeline-cpu`.
  - `report.json` differs only in `pred_file`.
  - `details.jsonl` differs only in timings and request ids.
  - Result: 0.7091 / 0.7015 / 0.8117, gap 0.0212, top-70 0.376, routed 28.2%.
- **`python experiments/routing/analyse_routing.py`** (exit 0, no API call).
  - These are byte-identical to the committed versions: `results_routing.json`, and in `dev-pipeline-llm-replay`, `predictions.jsonl`, `report.json`, `report_official.json` and `meta.json`.
  - `traces.jsonl` differs in timings, request ids and `policy_version`. The committed traces say `2026-10-14.1`, while the shipped policy is `.2`; the predictions are the same.
  - The rerun left the new traces in the working tree. I restored nothing, as instructed. The pre-run copies are in `snapshot_before_routing/`.
- **Live LLM against the recorded answers.** `dev-http-llm` and the replay differ in action on 64 of the 460 rows that called Luna. Net effect: 19 right against 18. A reviewer's rerun with the LLM on will move by a few tenths of a point.

## 7. Completeness against the problem and rubric

| Item | State |
|---|---|
| README.md | Present but stale. It says "Status: Phase 0", "HTTP service arrives in Phase 1" and "models/ … not committed" (`classical_v0.joblib` is committed). It has no failure modes or limitations, and no runbook. |
| EXPERIMENT_LOG.md | Dated, with dead ends. It lacks the submitted run (`dev-http-llm`), the encoder test, and the uncommitted ablation, chaos and taxonomy work. |
| DECISIONS.md | Two decisions; D2 covers an inconclusive test. D1 is still marked "provisional" and lacks an explicit initial hypothesis, the constraint that mattered, and links to raw runs. There is no coding-agent disclosure (P1-1 has one). |
| LANDSCAPE.md | One page, with sources. Stale: line 19 says "loadtest.py has yet to run with Luna"; line 9 says "no result yet"; it gives $0.03 per 1,000 against the measured $0.040. |
| MEMO.md | **Placeholder.** None of the required content: right, wrong, share routed and why, cost per 1,000, Monday checks, residual risk, who bears it, pause condition. |
| LICENSE | MIT. Allowed. |
| scripts/reproduce.sh | Present. Both reruns work. |
| results/manifest.json | Has `claimed` and 69 runs (70 after my run). **`claimed` is from the wrong run**; cost and latency are null. |
| results/raw/ | Non-empty, 70 runs. |
| dev_predictions, noisy_predictions, hostile_results, latency_bench | Present and consistent with `predictions.jsonl`. |
| Failure taxonomy of the baseline | Only R1 (reconstructed). The table of which intervention moved which category exists in scratch (`e1_compare.json`) and in the uncommitted `experiments/taxonomy/`, not in the committed docs. |
| Noise by kind | Given for v0 only (P0-3). For the shipped system on dev_noisy (intent / action, 50 rows each): code_mixed 0.70 / 0.76, empty 0.62 / 0.86, screenshot 0.50 / 0.84, two_intents 0.68 / 0.86, typo 0.72 / 0.76, very_long 0.68 / 0.82. |
| Hostile before / after | R10 (before) and one line in P1-1 (after). No dedicated section. |
| Working tree | 35 deletions, 12 modified files and 61 untracked paths, not committed. HEAD's manifest has 19 runs and the old ledger path. |

## Findings

See `audit.json` for the full list with severity, file and fix. The two critical items:
1. `results/manifest.json`: set `claimed` from `dev-http-llm`, and fill in cost and p95 latency.
2. `MEMO.md`: write it.
