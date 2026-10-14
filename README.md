# OP-06 Triage the Inbox

`POST /triage` reads up to eight turns of a customer-support conversation (ABCD) and returns the customer's
intent, the next action the written procedure permits, a confidence, and whether a person should take the case.
This file says how it decides, what it scored, how to run it and where it fails. `RUNBOOK.md` covers operation.

## How it decides

The request path is `triage/pipeline.py`; `docs/DESIGN.md` is the full contract.

1. **Gate** (`triage/gate.py`). Removes lone surrogates, zero-width and bidi characters, replaces control
   characters with spaces, applies NFKC, drops sentences repeated inside a turn, caps turn length and keeps the
   last 8 turns. It sets flags such as `injection`, `junk` and `no_dialogue` (no agent turn yet). A context with
   no letter or digit left gets the abstention envelope.
2. **CPU model** (`models/classical_v0.joblib`). Word and character TF-IDF over all 8 turns. A linear SVM picks
   the intent; a logistic regression picks the action from the text and the actions the agent has already logged.
3. **Policy prior and confidence.** Action probabilities are weighted by how likely each action is permitted
   under the predicted flow (`policy/policy.toml`). A second model, fit on held-out training rows, predicts
   whether the chosen action is right; its output is the confidence.
4. **LLM step** (optional). When CPU confidence is below 0.642841 (about 20% of dev rows) and no gate flag routes
   the row, GPT-6 Luna picks the next action from the closed list of 30, shown 8 similar training conversations.
   A usable answer replaces the action, and the confidence becomes the mean of the CPU confidence and 1 (same
   action) or 0 (different). The LLM never sets the intent or `needs_human`; any failure keeps the CPU answer.
5. **Routing.** `needs_human` is true when confidence is below the point for the action's risk tier (0.8883 for
   refunds, order and account changes, purchases, promo codes and passwords; 0.7104 otherwise), when the action
   is outside the flow's written procedure (`off_policy`), or when a gate flag routes. Flag-routed rows have their
   confidence capped at 0.5.

## Results

All numbers are on the published development files: dev (2,000 rows) and dev_noisy (300), plus the 9 hostile rows
in the schema check. Dev helped choose the system, so the hidden test will score lower. The cleanest estimate of
the LLM step's gain comes from 3,966 held-out training rows: action +0.025 [+0.017, +0.032] (EXPERIMENT_LOG P0-5).
Intervals are 95%, from a conversation bootstrap.

| Dev measurement | Submitted: LLM on, 2nd live run (`dev-http-llm`) | 1st live run (`dev-http-llm-r1`) | LLM off (`dev-pipeline-cpu`) | Bar | Baseline |
|---|---|---|---|---|---|
| Intent accuracy | 0.7091 | 0.7091 | 0.7091 | ≥ 0.55 | 0.3748 |
| Intent macro-F1 | 0.7015 | 0.7015 | 0.7015 | ≥ 0.50 | 0.4508 |
| Action accuracy | 0.8365 [0.8162, 0.8563] | 0.8365 [0.8149, 0.8569] | 0.8117 [0.7912, 0.8323] | ≥ 0.55 | 0.3913 |
| Action, clean / noisy | 0.8435 / 0.7900 | 0.8395 / 0.8167 | 0.8145 / 0.7933 | | 0.4010 / 0.3267 |
| Clean-to-noisy gap | 0.0535 | 0.0228 | 0.0212 | ≤ 0.10 | 0.0743 |
| Top-70% error ratio | 0.433 | 0.433 | 0.376 | ≤ 0.85 | 1.0 |
| Schema valid, 2,309 rows | 100% | 100% | 100% | 100% | 100% |
| Policy violation (gold: 0.0883) | 0.0839 | 0.0848 | 0.0804 | | 0.1404 |
| Sent to a person, 2,300 labelled rows | 24.3% | 24.2% | 28.2% | | |
| Action accuracy when not sent | 0.915 | 0.916 | 0.920 | | |
| LLM calls | 460 (19.9% of 2,309) | 460 | 0 | | |
| p50 / p95 over HTTP, 8 clients, M1 | 8.9 ms / 1,913 ms | 8.9 ms / 1,656 ms | 26 ms / 38 ms (`dev-http-cpu`) | | |
| LLM cost per 1,000 messages, uncached / cached | $0.125 / $0.040 | not kept / $0.040 | $0 | | |

Each graded run folder in `results/raw/` holds `report_official.json` (grader output), `report.json` (intervals,
routing, noise kinds) and, for HTTP runs, `latency_bench.json`. Two kinds differ: `policy-replay-*` hold
`report.json`, `report.txt` and `meta.json`, and `probe-2026-10-13/` is the archive of the 13 Oct probes, with its
own `README.md`. `results/manifest.json` claims `dev-http-llm`. The audit's own scorer gives the same numbers for
the submitted file (`results/audit/score_submitted.txt`). `dev-http-cpu` ran under an earlier policy version that
differs in routing only.

- **Why the second live run.** Its `service_log.jsonl` records each LLM call's tokens and cost; the first run kept
  only totals. It is not the better run: action accuracy is equal and its noisy gap is wider.
- **Spread.** Luna's answers varied between runs. The request sets no temperature or seed, so the provider's
  default sampling applies; no run tried temperature 0. 71 of 2,300 actions differ between the two live runs: 21
  right only in the second, 21 right only in the first, 29 wrong in both. That moves the gap on 300 noisy rows
  from 0.0228 to 0.0535. Expect a rerun to move as much.
- **Noisy gap.** This is the bar closest to failing: the submitted run's interval reaches 0.0972 against a bar of
  0.10. The LLM step widens the gap (0.0212 off; 0.0228 and 0.0535 in the live runs; 0.0338 replaying recorded
  answers) and gains nothing on noisy rows (0.79 against 0.7933 off). The grader's `matched_noisy_gap` for the
  submitted run is 0.0333 (`report_official.json`).
- **Noise kinds, submitted run** (intent / action, 50 rows each): code_mixed 0.70 / 0.76, empty 0.62 / 0.82,
  screenshot 0.50 / 0.78, two_intents 0.68 / 0.88, typo 0.72 / 0.72, very_long 0.68 / 0.78.
- **Hostile pack.** All 9 rows get a schema-valid 200 with `needs_human` true: 7 answered with confidence 0.5 or
  less, 2 (empty, emoji flood) abstain (`results/hostile_results.jsonl`).

## Running it

Needs bash, Python 3.12 and [uv](https://docs.astral.sh/uv/), on Linux or macOS. No GPU, no API key.

```sh
scripts/reproduce.sh setup              # locked dependencies into .venv (uv sync --frozen)
scripts/reproduce.sh verify             # data/op06 against data/SHA256SUMS
scripts/reproduce.sh test               # pytest, API keys removed from the environment
scripts/reproduce.sh serve              # the service on $PORT (default 8000), in the foreground
scripts/reproduce.sh smoke              # start, send 9 hostile + 20 dev rows + 5 malformed requests, check, stop
scripts/reproduce.sh eval my-run        # 2,309 rows in process, official grader, into results/raw/my-run/
scripts/reproduce.sh http-eval my-run   # 2,309 rows over HTTP at 8 clients, graded
URL=http://127.0.0.1:8000 scripts/reproduce.sh loadtest --concurrency 80 --out /tmp/lt.json
scripts/reproduce.sh train              # retrain beside the shipped model and compare (about 2 min on an M1)
scripts/reproduce.sh help               # every subcommand
```

```sh
curl -s localhost:8000/triage -H 'content-type: application/json' \
  -d '{"id": "x-1", "context": [{"speaker": "customer", "text": "where is my order?"}]}'
```

**What a rerun overwrites.** Without a run name, `eval` writes `results/raw/dev-pipeline-cpu/` (no key) or
`dev-pipeline-llm/`, and `http-eval` writes `dev-http-cpu/` (no key) or `dev-http-llm/`, the submitted run.
`http-eval` always rewrites `results/*.jsonl` and `results/latency_bench.json`; `loadtest` writes the latter
unless given `--out`. `eval` and `http-eval` rebuild the manifest. `git restore results/` undoes the overwrites;
delete any new `results/raw/<run>/` folder by hand.

**No key** (the default, `TRIAGE_LLM=auto`): the LLM step is off and the CPU model answers every request. The
audit's rerun, `TRIAGE_LLM=off scripts/reproduce.sh eval audit-cpu`, matched `dev-pipeline-cpu` byte for byte.

**With an LLM**, any OpenAI-compatible chat-completions endpoint works:

```sh
export TRIAGE_LLM_BASE_URL=https://your-gateway/v1      # default https://openrouter.ai/api/v1
export TRIAGE_LLM_API_KEY=...                           # or OPENROUTER_API_KEY
export TRIAGE_LLM_MODEL=openai/gpt-6-luna-20260922      # default; use your gateway's name for this snapshot
export TRIAGE_LLM_DIALECT=auto  # auto | openrouter | openai; auto = openrouter for openrouter.ai, else openai
scripts/reproduce.sh serve
```

The first start with the step on builds `models/retrieval.joblib` (about 15 s). Each call asks for a strict
JSON-schema answer and sends no temperature. `TRIAGE_LLM_DIALECT` sets the other field names. With a base URL that
is not `openrouter.ai`, `auto` picks `openai`, which sends only OpenAI Chat Completions names: `model`, `messages`,
`response_format` (strict JSON schema), `max_completion_tokens` 2048 and `reasoning_effort` `"none"`; no `usage`,
`provider`, `reasoning` or `max_tokens`. The `openrouter` dialect sends the tested body: `max_tokens`,
`reasoning: {"effort": "none"}`, `usage` and the `provider` pin. A gateway in front of OpenRouter that passes
OpenRouter's fields through should set `TRIAGE_LLM_DIALECT=openrouter`. Any other value stops start-up.

Only the `openrouter` dialect has run against a real provider (`dev-http-llm`, `dev-http-llm-r1`, both at
`openrouter.ai`); the `openai` dialect has run only against the local fake (`experiments/chaos/fake_upstream.py
--mode strict`, which answers 400 to any field outside that list). A gateway that rejects the fields it is sent
fails every call, and the service then gives LLM-off answers with no error to the caller; `smoke` counts a failed
call as an LLM call, so it still passes. So after a run, check `/healthz` `llm.dialect` and read `/metrics`: if
`llm.by_status` shows mostly `http_error` or `breaker_open`, or `/healthz` shows `llm.breaker` open, the run
measured the LLM-off configuration (dev action 0.8117, gap 0.0212, top-70 0.376, 28.2% routed;
`dev-pipeline-cpu`), not the claimed one. Costs are computed only for the model ids in `triage/llm_action.PRICES`.
With `TRIAGE_LLM=on` and no key, calls fail as `http_error`, the breaker opens and the CPU answer stands (code
reading; `chaos-error500` shows the same path). A full `eval` or `http-eval` with the step on makes about 460
calls, about $0.29 at uncached list price. `docs/DESIGN.md` lists every `TRIAGE_*` setting.

## Service contract and limits

`GET /healthz` returns 503 while loading, 200 when ready. `POST /triage` takes `{id, context, turn_index?}` and
returns `{id, intent, action, confidence, needs_human}`, echoing the id. A valid envelope with no readable text
gets a 200 abstention: `intent "unknown"`, `action "none"`, `confidence 0`, `needs_human true` and
`error: {"code": "invalid_input", "message": ...}`. Other failures return `{"error": {"code", "message"}}`:

| Case | Answer |
|---|---|
| Bad JSON; nesting over 64 levels | 400 `bad_json` |
| Not an object, or no non-empty string `id` | 400 `invalid_envelope` |
| Body over 1 MiB / not received within 10 s | 413 `body_too_large` / 408 `body_timeout` |
| Over 128 requests in the app; model loading | 503 `overloaded` / `loading`, `Retry-After: 1` |
| Over 256 open connections | plain-text 503 from uvicorn |

The largest published row is 7,104 bytes (`data/op06/dev_noisy.jsonl`). An LLM call has a hard 4 s deadline and no
retry; at most 8 wait at once, and a ninth keeps the CPU answer. After 5 failed calls in a row a breaker skips
calls for 30 s. A repeated body gets the cached answer, with no second LLM call.

## Models, prices and spend

| Role | Exact id | $ per million tokens: in / out / cache read / cache write | Price source |
|---|---|---|---|
| LLM step | `openai/gpt-6-luna-20260922` on OpenRouter, OpenAI host only, fallbacks off | 0.10 / 0.50 / 0.01 / 0.125 | OpenRouter `GET /api/v1/models`, retrieved 13 Oct 2026 |
| Intent step, tested, not shipped | `claude-haiku-5-5` | 0.10 / 0.50 / 0.01 / 0.125 | `experiments/llm_tail/llm_common.py` `PRICES`, from the claude-api skill's models.md / model-migration.md, retrieved 13 Oct 2026 |
| CPU model | `models/classical_v0.joblib`, SHA-256 `c007c7a3…` | none | |

We count Luna as budget class on price: it costs what Haiku 5.5 costs and less than the fixed `gpt-5-mini`
reference ($0.25 / $2.00). MODELS.md defines the class by tier (mini, nano, Flash, Haiku), and I have no
provider page that places Luna in a tier. Luna responses in the ledger name the alias `openai/gpt-6-luna`, which
the 13 Oct listing maps to the dated snapshot; MODELS.md says aliases alone are not reproducible pins. If Luna is
ruled out on either point, the configuration that remains is the CPU-only one (`TRIAGE_LLM=off`): dev intent
0.7091, macro-F1 0.7015, action 0.8117, gap 0.0212, top-70 0.376, schema 100%, $0 per message
(`dev-pipeline-cpu`), p95 38 ms over HTTP on the M1 (`dev-http-cpu`, older code and policy). Costs follow
MODELS.md: uncached list price, cached shown beside it. Dependencies are pinned in `pyproject.toml` and `uv.lock`.

| Measured API spend at list price | Calls | With caching | Uncached |
|---|---|---|---|
| LLM experiment runs in `results/raw/` (ledger) | 10,646 | $2.68 | $8.71 |
| The two live service runs, warm-up included | 926 | $0.19 | $0.29 for `dev-http-llm`; not kept for `-r1` |
| 13 Oct probes, before this repo (`results/raw/probe-2026-10-13/llm/ledger.jsonl`) | 3,514 | $1.58 (OpenRouter part $1.15 at list; it billed $1.29) | $3.33 |

Sources: `results/manifest.json` (`spend_usd`, `ledger`, and the `probe-2026-10-13` entry's `probe_spend_usd`)
and AUDIT.md section 5, which agree. At list price the total is about
$4.45 with caching ($4.59 if the probe's OpenRouter part is counted as billed) and $12.33 uncached for the
recorded parts. The first live run's uncached cost was not recorded; with the audit's estimate of about $0.29 for
it, the uncached total is about $12.62 (EXPERIMENT_LOG, Spend). The ceiling is US$50.

## Hardware

An **Apple M1 laptop** (8 cores, 8 GB, Python 3.12.4) ran the submitted run and every other graded run except
`linux-retrained-cpu`, all 14 Oct LLM runs and the fault-injection runs. A **shared Linux box** (2× Xeon Gold
6330, Python 3.12.15, load average 24 to 38 on 112 cores from other jobs) ran the clean-machine reproduction, the
retrain check, the CPU-only load tests, the thread-scaling check (`linux-threads`) and the encoder fine-tune
(`encoder-ett32A`); its latency is an upper bound for a quiet machine. The 13 Oct probes used both. No GPU.
Compute is outside the API budget.

## Known failure modes and limitations

- **Dev chose the system.** The CPU configuration and the LLM share were picked while looking at dev.
- **The noisy gap is the bar most at risk** on the hidden test: its dev interval reaches 0.0972 against 0.10, and
  the LLM step widens it (Results).
- **Luna's answers vary between runs** at the provider's default temperature (71 differing actions between two
  runs). Against a replay of recorded answers, the first live run differed on 64 of 460 LLM rows (AUDIT.md) and
  the submitted run on 65 (EXPERIMENT_LOG P3-5).
- **The 8-turn window hides the opening of long conversations.** Of 564 clean intent errors left, 224 have the
  opening outside the window and 277 pick the wrong intent within the right flow
  (`experiments/taxonomy/taxonomy.md`). A procedure rule about earlier steps cannot see past the window.
- **Noise weak spots.** Screenshot descriptions get the intent right half the time; typo rows get the action
  right 72% of the time. The service returns one intent: a second request sets a `second_intent` flag (19 of 50
  `two_intents` rows) that is logged, not routed (EXPERIMENT_LOG P2-1).
- **Confidence orders errors less well with the LLM step** (top-70 ratio 0.433 against 0.376). A provider that
  returns wrong but well-formed actions passes every check; each changed action goes to a person, but no counter
  shows the fault (`experiments/chaos/chaos.md`).
- **Injected text still moves labels** in 20% to 27% of conversations it is added to; the `injection` flag routes
  all of them. Reworded triggers ("disregard everything above") are not detected.
- **One worker.** The CPU path served 84 req/s at 8 clients on the busy Linux box and 302 req/s on the M1. The LLM
  step tops out near 5 calls/s per process; past that, most LLM-gated rows keep the CPU answer.
- **Idle connections.** From reading uvicorn's code, not tested: it has no header timeout, so 256 silent
  connections would block new ones. Put a proxy with header and body timeouts in front.
- **Retraining differs across platforms.** A Linux retrain changed 5 of 2,309 actions (EXPERIMENT_LOG P3-1), so
  the model file ships and `train` only compares (it fails above 1% of rows). The retrieval index is rebuilt,
  not shipped. Before a tie-order fix a Linux rebuild matched the tested examples on 198 of 200 rows; after it,
  Linux and M1 gave the same examples on 2,300 of 2,300 rows (EXPERIMENT_LOG P3-4, no raw file kept), and 4 rows
  differ from the tested examples only in the order of tied examples. That run also failed a timing test (0.545 s
  against 0.5 s), since changed.
- **No clean-machine run of the final code.** The Linux run from `git archive` (P3-1) came before the retrieval
  fix (18:41) and before `triage/pipeline.py`, `service.py` and `llm_action.py` changed (18:52 to 19:02); no log
  records the test suite passing on the final code. `dev-http-cpu` (p95 38 ms) also ran on older code and policy.
- **Luna returns an alias id.** All 6,276 Luna calls in `results/raw/llm-ledger/ledger.jsonl` (3,966 fold-B,
  2,300 dev, 10 smoke) asked for `openai/gpt-6-luna-20260922` and got back `openai/gpt-6-luna`, so no response
  proves which snapshot served it. The service log now records `llm_model`, `llm_provider` and `llm_response_id`
  for each call; the two live runs came before this and did not record them.
- **A 429 is an ordinary failure.** The service does not read `Retry-After` or retry: the CPU answer stands and
  the failure counts toward the breaker (tested only against the fake's `ratelimit` mode).
- **Not tested:** memory over hours, several workers.

## Repository layout

| Path | What |
|---|---|
| `data/op06/` | Published OP-06 files, unchanged; sums in `data/SHA256SUMS` |
| `triage/` | The service: gate, CPU model, policy, prompts, retrieval, LLM step, pipeline, HTTP |
| `policy/policy.toml` | Permitted actions per flow, risk tiers, routing points, rules; reloaded on save |
| `models/` | `classical_v0.joblib` ships; `retrieval.joblib` and `retrained/` are built locally |
| `harness/`, `scripts/`, `tests/` | Grading and manifest; `reproduce.sh` and tools; unit and HTTP tests |
| `experiments/` | One folder per question, with pre-registrations for `llm_tail/`, `llm_confirm/` and `encoder/` |
| `results/` | Submitted predictions, `latency_bench.json`, `manifest.json`, `raw/<run>/` for every run, the LLM ledger of every API attempt, the 13 Oct probe archive (`raw/probe-2026-10-13/`), `audit/` |

`MEMO.md` is the page for the support manager. `DECISIONS.md` gives the choices, the options rejected and what
would reverse them. `EXPERIMENT_LOG.md` is dated, dead ends included; its 13 Oct entries are reconstructed.
`LANDSCAPE.md` covers what already exists. `RUNBOOK.md` is for whoever runs the service. `docs/DESIGN.md` has the
interfaces, settings and limits. `results/audit/AUDIT.md` is an audit run by a separate coding-agent session with
its own scorer; it read the first live run under its old name `dev-http-llm` (now `dev-http-llm-r1`).

## Coding-agent use

Coding agents wrote most of the code from the contract in `docs/DESIGN.md`, ran most experiments, and drafted
these documents, this README included, from the run files. A second agent tried to refute each review finding
before it was fixed (EXPERIMENT_LOG P1-1), and the audit in `results/audit/`, another agent session, recomputed
the headline numbers with a scorer written from the problem statement. `DECISIONS.md` says which outputs were checked by hand.
