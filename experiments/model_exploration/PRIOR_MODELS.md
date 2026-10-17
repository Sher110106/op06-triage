# Prior neural-model and LLM comparisons

This is an inventory of existing results before the current model exploration.
The archived 13 October probe is reconstructed and unaudited; later comparisons
are still on repeatedly inspected public dev or old train-fold partitions. None
of those is fresh confirmation for a new model. The current 65/10/10/15 grouped
experiment must use its own fit, selection, calibration and confirmation IDs.

## Encoders and embeddings

| Model or representation | Existing evidence | Status and reusable artifacts |
|---|---|---|
| `jhu-clsp/ettin-encoder-32m` | Initial 13 Oct probe: intent .733, macro-F1 .716, clean/noisy action .834/.793; 25 min CPU, but on the probe's scorer/training split (`EXPERIMENT_LOG.md` R8; `results/raw/probe-2026-10-13/encoders/logs/ft_ett32.json`). | Not a final comparison. Re-run E-1 used fold A (14,313 fit rows, 1,721 early-stop rows), two epochs on 16 CPU threads. Probability blend chosen on dev: action .5 classical/.5 encoder, intent .3/.7. On old fold B, encoder alone was +.0089 action [−.0005,+.0186] vs classical; blend +.0273 [+ .0203,+.0341]; vs Luna +.0026 [−.0053,+.0098], missing the −.005 floor by .0003. Rejected for deployment (`experiments/encoder/PREREGISTRATION.md`, `experiments/encoder/results_encoder.json`, `EXPERIMENT_LOG.md` E-1, `DECISIONS.md` D2). |
| `intfloat/multilingual-e5-small` fine-tune | Intent .348, action .690 on dev; likely setup fault, not evidence the family is intrinsically weak (`results/raw/probe-2026-10-13/encoders/logs/ft_e5s.json`). | Negative probe only; no successful follow-up. |
| `BAAI/bge-small-en-v1.5` frozen embeddings | TF-IDF + BGE-small on all 8 turns: intent .729 clean/.6733 noisy, vs TF-IDF .7215/.6333. BGE alone LR .694/.6533; kNN weaker (`results/raw/probe-2026-10-13/encoders/logs/b1_results.jsonl`). | Small noisy-intent gain; not an action model and no independent confirmation. |
| `minishlab/potion-base-8M` frozen embeddings | TF-IDF + Potion all-8 intent .714/.610 noisy; Potion alone .670/.523 (`b1_results.jsonl`). | Did not beat the sparse baseline. |
| Other embedding encoders | Runtime metadata only for `BAAI/bge-base-en-v1.5`, `intfloat/multilingual-e5-base`, and `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (`results/raw/probe-2026-10-13/encoders/logs/embed_meta_*.json`). | No comparable classifier accuracy recorded. |

E-1 training began before its pre-registration commit. Its fold-B comparison also
retained classical confidence instead of recalibrating the selected ensemble, so
it does not establish deploy-ready confidence or routing quality.

The retained Ettin artifacts are temperature-scaled probability arrays and row IDs
for `foldB`, `dev`, `noisy`, and `hostile`, plus labels and logs, under
`results/raw/encoder-ett32A/`. There is no checkpoint, tokenizer bundle, seed, or
recorded Hugging Face revision. These arrays can support offline scoring or
reweighting on those same previously used rows; they cannot train a new model or
provide independent confirmation. The frozen-embedding archive has aggregate
metrics and model metadata, not cached embedding matrices, so alternate heads or
fusion weights require re-encoding.

## LLM comparisons

The 13 October probe compared Haiku 5.5, GPT-6 Luna, Qwen3.8 27B, Gemma 4 31B,
Gemini 3.5 Flash-Lite, and Solar Mini 4 on 360 labelled examples (240 clean,
120 noisy); Gemini 3.8 Flash had only a 60-row run. In that sample no LLM beat
the CPU action model. V3 action deltas included Luna −.022 [−.064,+.017], Haiku
−.058 [−.100,−.017], Gemma −.050 [−.094,−.008], Flash-Lite −.072
[−.117,−.031]; intent gains for several models did not translate to action.
Qwen had provider-host failures and needed a host-pinned redo. See
`results/raw/probe-2026-10-13/llm/report_tables.md` T1–T4 and
`results/raw/probe-2026-10-13/README.md`.

The later pre-registered full-dev test evaluated the same V3 prompt on 1,940
fresh rows after excluding the 360 probe IDs. Luna on the old 34.7%-routed tail
added +.014 action [−.001,+.029], below its +.020 rule. Haiku intent gained
macro-F1 +.031 [+.004,+.054] but accuracy was −.003 [−.024,+.018] and action
did not change. A dev-selected 20% Luna tail later passed old fold-B confirmation
(3,966 clean rows): +.0247 action [+.017,+.032] below confidence .642841.
Intent deltas were +.0144 Luna and +.0129 Haiku, below the .020 rule. Sources:
`experiments/llm_tail/PREREGISTRATION.md`, `experiments/llm_tail/results.json`,
`experiments/llm_confirm/PREREGISTRATION.md`,
`experiments/llm_confirm/results_B.json`, and `EXPERIMENT_LOG.md` P0-4/P0-5.

The Luna fold-B and dev raw calls and final HTTP predictions are cached, so the
exact old prompts can be rescored without another API call. They do not test a
new provider/model, and fold-B retrieval came from the old fold A. Do not treat
these responses as fresh evidence for the current grouped split. The current
final Luna run is `results/raw/dev-http-llm-20261017/`; it requested the dated
snapshot but the provider returned an alias, so exact snapshot identity remains
unverified.

## Worth testing, not already measured

- `microsoft/deberta-v3-small` and `answerdotai/ModernBERT-base` occur only as
  example commands in `experiments/encoder/ft_foldA.py`; no scored run or saved
  checkpoint was found. A multi-task encoder with explicit dialogue-state
  features is the most direct next comparison, especially if GPU training is
  available. Measure batch-1 CPU latency/memory as well as accuracy if CPU
  serving remains a requirement.
- A fresh BGE-small sparse/dense fusion or fine-tune could test the observed
  noisy-intent signal. Its old vectors are absent, so it requires re-encoding.
- SetFit and the ABCD release's BERT action models were not run. The old landscape
  notes that the release models were skipped under a no-GPU assumption; hardware
  is now open. FastText was also not tried. A budget-class generation model on a
  new uncertain-tail split is possible, but the older prompt sweeps provide no
  action advantage and should be treated only as background.

OP-06's published model constraints still define budget-class candidates as
small hosted mini/nano/Flash/Haiku models or open-weight models up to about 35B;
model IDs, revisions, providers, and public price sources must be pinned in the
submission (`hiring-problems/MODELS.md`).
