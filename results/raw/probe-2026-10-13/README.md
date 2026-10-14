# 13 Oct 2026 research probes (archive)

These are copies of the files the 13 Oct 2026 development probes left in the research workspace (`scratch/` and
`research/`, beside this repo). The probes ran before this repo existed, on a remote 112-core CPU box and on an M1
laptop. I copied the files on 14 Oct so that every probe number quoted in `EXPERIMENT_LOG.md` (R1 to R10) and
`LANDSCAPE.md` has a file in the repo.

**Unaudited.** No one reran these probes, and the code that made them is not here. The audit
(`results/audit/AUDIT.md`, section 2) read the same files in `scratch/` and matched some of the quoted numbers.
That checks the quotes, not the runs. The repo's own reruns are the P-entries (P0-3, P4-1, P4-2).

- **Copies are exact.** Each file matches its source byte for byte, with its 13 Oct file time kept. `SHA256SUMS`
  gives the SHA-256 of all 172 copied files (`shasum -a 256 -c SHA256SUMS` from this folder checks them).
- **No API keys.** A scan on 14 Oct of every file for `sk-or-v1-` followed by 40 or more hex characters, and for
  `sk-ant-` followed by 20 or more characters, found none. No file holds any `sk-or-`, `sk-ant-` or `sk-proj-`
  string. `llm/keycheck.jsonl` holds only the OpenRouter key's usage and limit, not the key.
- **Same data.** The probes read train, dev, dev_noisy and hostile from `hiring-problems/references/OP-06/`. Those
  four files have the same SHA-256 as `data/op06/` here.
- **Nothing left out.** The chosen sources come to 9.0 MB, under the 20 MB limit, so every file is here. Not
  chosen, and still only in `scratch/`: the probe scripts, the pickled features and models in
  `probe_classical/out/` (about 250 MB), the prediction files in `probe_noise/out/*.jsonl`,
  `probe_llm/{cpu_preds,gold,grader_preds,prompts,explore}/`, and the plain-text `.log` files in
  `probe_encoders/logs/`. `llm/llm-probe.md` names some of these; they are not here.

## Folders

Paths map simply: `scratch/probe_<name>/<path>` is `<name>/<path>` here, and `research/llm-probe.md` is
`llm/llm-probe.md`. So `scratch/probe_classical/out_remote/e6_noise_final.json`, which R6 cites, is
`classical/out_remote/e6_noise_final.json`.

| Folder | Files | Source | What it holds |
|---|---:|---|---|
| `llm/` | 77 | `scratch/probe_llm/`: `runs/`, `scores/`, `meta/`, `grader_reports/`, `run_summaries.jsonl`, `ledger.jsonl`, `keycheck.jsonl`, `sample.json`, `retrieval.json`, `report_tables.md`; and `research/llm-probe.md` | The LLM probe: six budget LLMs on 369 rows (240 clean, 120 noisy, 9 hostile), and Gemini 3.8 Flash on 69 of them. Raw answers per run (`runs/`), scores (`scores/`), official grader reports (`grader_reports/`), one line per API attempt (`ledger.jsonl`, 3,514 lines), OpenRouter balance checks (`keycheck.jsonl`), the sample (`sample.json`), retrieved examples (`retrieval.json`), the OpenRouter model list and leaderboard notes (`meta/`), the write-up (`llm-probe.md`) and its tables (`report_tables.md`). |
| `classical/` | 45 | `scratch/probe_classical/out_remote/*.json` (44) and `out/e2_views.log` (1) | The classical-model probe on the remote box: baseline taxonomy (`e1_*`), intent models (`e2b_intent.json`, `out/e2_views.log`), action models (`e4_action.json`, `e5__*`), confidence (`report_*`), noise (`e6_*`), augmentation and de-duplication (`e8_*`), latency and bundle size (`e9_*`), the final grade (`final_grade.json`, `report_final_*`). |
| `noise/` | 39 | `scratch/probe_noise/out/*.json` | The noise probe: each noise kind described (`n1_*`, `perkind_all.json`), cleaning variants (`exp_n0_raw` to `exp_n7_dedup300`), per-kind runs (`exp_k_*`), ensembles and final variants with their grader reports (`final_*`, `report_*`), bootstrap comparisons (`boot_*`). |
| `encoders/` | 11 | `scratch/probe_encoders/logs/*.json`, `*.jsonl` | The encoder probe: frozen embeddings with TF-IDF by view (`b1_results.jsonl`), the hostile rows after cleaning (`b4_hostile.json`), embedding settings (`embed_meta_*`), fine-tuned multilingual-e5-small (`ft_e5s.json`) and Ettin-32M (`ft_ett32.json`), the label list. |

`out/e2_views.log` is the one file from outside the four listed sources: R2 and the status table after R10 name it.

## Which file holds each quoted number

| Entry | Numbers | File here |
|---|---|---|
| R1 | 1,228 intent errors (518 / 408 / 235); 1,198 action errors (460 / 283 / 225); 94 off-policy + 50 alias | `classical/out_remote/e1_baseline_taxonomy.json` |
| R1 | Baseline 0.3748 / 0.4508 / 0.3913 | `classical/out_remote/baseline_report.json` |
| R2 | Clean dev intent 0.718 (LinearSVC, C=0.3, all turns) | `classical/out/e2_views.log` |
| R2 | Last 4 turns 0.515, last customer turn 0.226 | `encoders/logs/b1_results.jsonl`: TF-IDF with logistic regression, views `last4` (0.5145) and `cust` (0.226). These come from the encoder probe, not from `e2_views.log`. |
| R2 | Opening outside the window in 60.4% of rows; intent 0.628 there, 0.856 elsewhere | `classical/out_remote/e1_tax_final.json` (0.604, 0.6275, 0.8561) |
| R4 | Action 0.808; wrong flow 114 → 64; state ignored 10 → 81; violations 0.080 / 0.045, gold 0.088 / 0.057 | `classical/out_remote/e1_compare.json`, `final_grade.json`, `e5__a_direct_text+state_LR_C10.0__soft_alias.json` (0.8078); the other variants in `e4_action.json` and `e5__*` |
| R5 | Threshold 0.8498; 34.7% routed; 2,280 distinct confidences | `classical/out_remote/e5__a_direct_text+state_LR_C10.0__soft_alias.json` (0.8498); `final_grade.json` (0.3465 of dev + noisy); `report_final_*.json` (2,280) |
| R6 | very_long intent 0.16 → 0.68; action gap 0.050 → 0.024 | `classical/out_remote/e6_noise_final.json`; `report_final_t1_nodedupe.json` against `report_final_t1.json` |
| R6 | The pair withdrawn on 14 Oct (0.10 → 0.68); NFKC and character-collapse variants | `noise/out/exp_n0_raw.json`, `exp_n5_dedup600.json`, `exp_n1_nfkc.json`, `exp_n2_nfkc_rep.json` |
| R6 | Noise augmentation | `classical/out_remote/e8_aug.json` |
| R7 | 0.7091 / 0.7015 / 0.8104 / 0.0235 / 0.3734 | `classical/out_remote/final_grade.json`, `report_final_t1.json` |
| R7 | p50 8.02 ms, p95 14.62 ms (2 threads); bundle 72,198,900 bytes, 35,338,775 compressed | `classical/out_remote/e9_latency_t2.json` |
| R8 | Ettin-32M 0.733 / 0.716 / 0.834 / 0.793, 25.1 minutes | `encoders/logs/ft_ett32.json` |
| R8 | multilingual-e5-small intent 0.348 | `encoders/logs/ft_e5s.json` |
| R8 | bge-small with TF-IDF +0.007 clean, +0.040 noisy intent; kNN weaker than logistic regression | `encoders/logs/b1_results.jsonl`: all 8 turns, fused 0.729 against TF-IDF 0.7215 clean, 0.6733 against 0.6333 noisy; the `intent_knn_*` fields |
| R9 | Luna −0.022 [−0.064, +0.017] action; intent +6 to +7 points | `llm/report_tables.md`, `llm/llm-probe.md` |
| R9 | Cascade; stated confidence; leaderboards and Spearman; Qwen hosts | `llm/llm-probe.md` (cascade table T7, section 7, section 6.4); `llm/meta/leaderboards_2026-10-13.md`; `llm/runs/qwen38_V3*.jsonl` |
| R9 | Spend: Anthropic $0.4318, OpenRouter $1.2858 billed; 5 OpenRouter HTTP errors | `llm/ledger.jsonl` (1,412 Anthropic and 2,102 OpenRouter attempts; 5 HTTP 400s) |
| R10 | hostile-002, 003 and 006 confident and not routed; 007 and 008 abstain | `classical/out_remote/e9_latency_t2.json` |
| LANDSCAPE, Encoders | bge with TF-IDF +0.007 / +0.040; potion tried; e5 0.348 | `encoders/logs/b1_results.jsonl`, `embed_meta_potion-8m.json`, `ft_e5s.json` |
| LANDSCAPE, Budget LLMs | Luna −0.022 [−0.064, +0.017], 360 rows | `llm/report_tables.md` |
| LANDSCAPE, Confidence | LLM stated confidence ordered errors poorly | `llm/llm-probe.md` |
| LANDSCAPE, ABCD release | A rule built from kb.json scored below a table learned from train | No file. No probe output in `scratch/` holds this figure. |

Not found in any file: the R3 parser rules and how they were checked (scripts only).
