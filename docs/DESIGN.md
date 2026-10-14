# v1 service design: the build contract

This file fixes the module boundaries and interfaces so the parts can be built separately and still fit.
Measured facts behind each choice live in `EXPERIMENT_LOG.md`; the choices themselves in `DECISIONS.md`.

## Request flow

```
HTTP POST /triage
  service.py   transport: load shedding (503), body limit and read deadline (413 / 408), JSON parse, envelope
               check (400), request id, idempotency cache, timing, metrics, logs
  pipeline.py  Triage.handle(obj, request_id) -> (response, trace); never raises
    gate.clean(obj)                      -> GateResult (cleaned turns, flags, abstain reason)
      abstain_reason set                 -> abstention envelope, stop
    classical raw probabilities          -> pa_raw (actions), pi (intents), pf (flows)
    policy prior                         -> pa = normalise(pa_raw * (pf @ P)), P from the policy file
    intent = argmax pi; action = argmax pa; confidence = stacker(pa, pi, pf, state)   (as classical v0)
    second-intent check                  -> flag "second_intent" (intent model on the last customer turn)
    LLM action step (if enabled, confidence < gate, and no ROUTE flag)
                                         -> action := LLM action when usable;
                                            confidence := (confidence + agree) / 2
    policy rules on the final action     -> flags "off_policy", "rule:<name>"
    routing                              -> needs_human, confidence cap for flagged rows
  response {id, intent, action, confidence, needs_human} (+ error object on abstention only)
```

The CPU path must reproduce `dev-classical-v0` exactly when the LLM step is off and the policy file equals
the bundle's table, with both thresholds equal to the bundle's `route_threshold` (`scripts/build_policy.py`
writes it in full): same intent and action on all 2,309 rows, and same confidence and `needs_human` on every row
not routed by a flag. A row routed by a flag may change `needs_human` and has its confidence capped at
`TRIAGE_FLAG_CONF_CAP` (0.5). With the default settings two kinds of flag route: a gate ROUTE flag, and
`off_policy` (`TRIAGE_ROUTE_OFF_POLICY=1`, decision D1 in `DECISIONS.md`). Measured on dev + dev_noisy against
`dev-classical-v0`: 49 rows with no gate ROUTE flag differ only through `off_policy` (confidence capped on all 49,
`needs_human` false to true on 25), and the 2 `no_dialogue` rows (4432-4 and its noisy copy) are routed and
capped. With `TRIAGE_ROUTE_OFF_POLICY=0` the 49 rows match v0 exactly.

## Modules and interfaces

### `triage/gate.py` (input validation and cleaning)

```python
ROUTE_FLAGS: frozenset[str]   # flags that force needs_human and cap confidence

@dataclass(frozen=True)
class GateResult:
    id: str
    context: list[dict]        # cleaned turns: {"speaker": "customer"|"agent"|"action", "text": str}, at most 8
    turn_index: int            # given whole number >= 0 (22.0 counts, as in JSON Schema), values of 10,000 and up
                               # cut to 9,999 (the model's features stop at 60); else len(context)
    flags: frozenset[str]
    abstain_reason: str | None # not None -> return the abstention envelope

def envelope_error(obj) -> str | None   # None if obj is a JSON object with a non-empty string "id" (any length,
                                        # any characters); else a message (-> HTTP 400)
def clean(obj: dict) -> GateResult      # never raises for any JSON value under the envelope
```

Cleaning order, per turn text: lone surrogates removed; C0/C1 control characters except `\t \n` replaced by
a space; runs of combining marks cut to 8 (NFKC sorts a run in time that grows with the square of its length:
8 turns of 16,000 marks took 1.4 s before the cut, 4 ms after); Unicode format characters (category Cf:
zero-width, bidi embeddings and overrides, BOM) removed; NFKC; repeated sentences dropped (the `dedupe_text` rule
from v0, turns of 200+ characters); length capped at `TURN_CAP` characters (keep the head). Non-string text
becomes "". Unknown speakers become "customer". More than 8 turns: keep the last 8.

Flags (strings): `truncated_turns`, `turn_capped`, `marks_capped`, `controls_removed`, `format_chars_removed`,
`bidi_override`, `surrogates_removed`, `bad_speaker`, `bad_text_type`, `bad_turn`, `bad_turn_index`,
`empty_turns`, `injection`, `no_dialogue`, `no_known_text`, `junk`.
`ROUTE_FLAGS = {injection, no_dialogue, no_known_text, junk, controls_removed, bidi_override, surrogates_removed}`.
- `bad_turn_index`: `turn_index` given but not a whole number >= 0 (replaced by len(context)), or 10,000 or more
  (cut to 9,999). Not a ROUTE flag; it shows the change in the trace.
- `no_dialogue`: no agent turn and no action turn (2 of 2,300 dev + dev_noisy rows: 4432-4, three customer turns
  with gold action search-faq, and its noisy copy; 12 of 20,000 train rows). This routes a real, if rare,
  conversation shape: a customer who writes several times before the agent answers.
- `injection`: instruction-like patterns in customer or agent text (for example "ignore ... instructions",
  "system:", "</context>", "new instruction", "needs_human", an action label such as "offer-refund" typed in a
  customer turn). The patterns run on the cleaned text and on a folded copy the model never sees: format
  characters become spaces (so "Ignore<ZWSP>all<ZWSP>previous<ZWSP>instructions" still has word breaks), marks
  are dropped after NFKD (CGJ, variation selectors, accents), Hangul fillers and the blank braille cell become
  spaces, Cyrillic and Greek look-alikes become Latin, and every dash becomes "-". Measured false positives on
  train + dev + dev_noisy (22,300 rows): 0. Still missed on purpose (pinned by tests as normal text): "needs
  human" with a space, a comma between "ignore" and "instructions"; also not covered: reworded triggers such as
  "disregard everything above" and a trigger split over two turns.
- `no_known_text`: the customer and agent turns hold letters, but under half of them are Latin (ASCII or Latin
  with accents). The model has no words for such text, so its answer comes from turn counts alone (8 turns of
  Chinese came back 0.854, unrouted). Measured on train + dev + dev_noisy: 0 rows.
- `junk`: a turn whose letters are under half of its non-space characters (length 8+), after cleaning.
- `abstain_reason`: no turn has a letter or digit after cleaning (empty input, emoji floods, pure junk).

### `triage/policy.py` and `policy/policy.toml`

The policy file is the only place the manager edits. TOML, read with `tomllib` (no extra dependency).

```toml
version = "2026-10-14.1"
alias = { "membership-privileges" = "membership" }
[flows]                         # flow -> permitted actions, as published, before alias
account_access = ["pull-up-account", ...]
[risk]                          # action -> tier; tiers set the routing threshold
high = ["offer-refund", "update-order", "update-account", "make-purchase", "promo-code", "make-password"]
[thresholds]                    # route when confidence < threshold for the action's tier (Phase 2 sets values)
high = 0.8498168103175511       # build_policy.py writes the bundle's route_threshold in full
low = 0.8498168103175511
[[rules]]                       # optional procedure rules
name = "refund-needs-verification"
action = "offer-refund"
requires_any_observed = ["verify-identity", "validate-purchase"]
effect = "route"                # "route" (flag and route) or "mask" (remove the action from candidates)
```

```python
@dataclass(frozen=True)
class Policy:
    version: str
    alias: dict[str, str]
    flow_permitted: dict[str, frozenset[str]]   # alias applied
    risk: dict[str, str]                        # action -> tier ("low" when unlisted)
    thresholds: dict[str, float]
    rules: tuple[Rule, ...]
    def prior_matrix(self, flows: Sequence[str], actions: Sequence[str]) -> np.ndarray  # float32, flows x actions, 1/0
    def masked_actions(self, observed: Sequence[str]) -> frozenset[str]                # actions removed by "mask" rules
    def violations(self, action: str, flow: str, observed: Sequence[str]) -> list[str] # "off_policy", "rule:<name>"
    def threshold_for(self, action: str) -> float

def load(path) -> Policy                   # validates action and flow names; raises PolicyError with a clear message
class PolicyStore:                         # thread-safe; current() reloads when the file's mtime changes,
    def __init__(self, path): ...          # checked at most every 2 s; a bad file keeps the last good policy
    def current(self) -> Policy: ...       # and records the error for /healthz
```

`scripts/build_policy.py` writes `policy/policy.toml` from the published `guidelines`-derived permitted lists in
train (identical to the v0 bundle table), with both thresholds set to the bundle's `route_threshold` written in
full (a rounded 0.8498 would route confidences in [0.8498, 0.8498168) differently from v0). `scripts/replay_policy.py --policy NEW.toml` re-scores dev + noisy
with a changed policy and reports how many actions, routes and violations change (the procedure-change drill).

### `triage/prompts.py`, `triage/retrieval.py`, `triage/llm_action.py` (the LLM action step)

- `prompts.static_prompt()` returns the exact V3 static prompt; its SHA-256 must equal
  `3511a0a0072bdcbfa89cd46bd66061e6bfdcd452d70ab2555d0f47657db58244` (a test asserts it).
  `prompts.dynamic_prompt(row, examples)` builds the V3 user message (8 examples, nonce markers, state hint).
  Port both unchanged from `experiments/llm_tail/llm_common.py`.
- `retrieval.Retriever`: word + char TF-IDF over train contexts, top 8 rows with at most one per conversation,
  exactly as `experiments/llm_tail/build_inputs.py`. `build(rows)`, `save(path)`, `load(path)`, `top_k(row, k=8)`.
- `llm_action.LLMConfig.from_env()`, `LLMActionStep(cfg, retriever, train_by_id).decide(row) -> LLMResult`; never raises.

Environment (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `TRIAGE_LLM` | `auto` | `on`, `off`, or `auto` (on when an API key is set) |
| `TRIAGE_LLM_BASE_URL` | `https://openrouter.ai/api/v1` | any OpenAI-compatible chat-completions endpoint, e.g. a reviewer gateway |
| `TRIAGE_LLM_API_KEY` | value of `OPENROUTER_API_KEY` | never logged, never returned |
| `TRIAGE_LLM_MODEL` | `openai/gpt-6-luna-20260922` | exact snapshot |
| `TRIAGE_LLM_PROVIDER` | `openai` | OpenRouter host pin; sent only when the base URL is OpenRouter (with fallbacks off) |
| `TRIAGE_LLM_TIMEOUT_S` | `4.0` | hard per-call timeout; no retries inside a request |
| `TRIAGE_LLM_GATE` | `0.642841` | call the LLM only when CPU confidence is below this |
| `TRIAGE_BUNDLE` | `models/classical_v0.joblib` | the CPU model bundle |
| `TRIAGE_POLICY_PATH` | `policy/policy.toml` | the policy file, reloaded on save |
| `TRIAGE_SECOND_INTENT_P` | `0.6` | flag `second_intent` when the last customer turn alone points to another flow with this probability |
| `TRIAGE_ROUTE_SECOND_INTENT` | `0` | `1`: `second_intent` routes too |
| `TRIAGE_ROUTE_OFF_POLICY` | `1` | `1`: `off_policy` routes (decision D1) |
| `TRIAGE_FLAG_CONF_CAP` | `0.5` | confidence cap for rows routed by a flag |

A bad value in any of these stops start-up: `scripts/reproduce.sh` checks them, and the policy file, before it
starts the server; the uvicorn app exits with status 3 if the model, policy or settings fail to load.

Call parameters match the tested configuration: JSON-schema response format (strict, enum-constrained),
reasoning effort `none`, `max_tokens` 2048, no temperature. A circuit breaker opens after 5 consecutive
failures and stays open for 30 s (calls skipped, CPU answer kept). At most 8 calls wait on the provider at once;
a ninth gets `busy` at once and keeps the CPU answer, so a slow provider cannot hold all 16 service workers (a fake
provider answering in 3.5 s cut CPU-only rows to 3.4 s p95 at 48 clients before this cap). A reply over 64 KB or
an answer over 16,384 characters is `invalid`; the lenient parse reads from the first "{" to the last "}" in
linear time. Token counts outside 0 to 10,000,000 (or cache counts above the prompt count) give no cost.
`LLMResult` carries `status`
(`ok`, `timeout`, `http_error`, `invalid`, `abstained`, `breaker_open`, `busy`, `disabled`), `intent`, `action`,
`latency_s`, `cost_usd` (list price: in 0.10, out 0.50, cache read 0.01, cache write 0.125 USD per million
tokens, OpenRouter listing retrieved 13 Oct 2026), token counts, and the returned model and provider.

### `triage/pipeline.py`

```python
class Triage:
    @classmethod
    def from_env(cls) -> "Triage"            # bundle, policy store, retriever, LLM config; TRIAGE_BUNDLE, TRIAGE_POLICY_PATH
    def handle(self, obj: dict, request_id: str) -> tuple[dict, dict]   # (response, trace); never raises
    def handle_many(self, objs: list[dict]) -> list[tuple[dict, dict]]  # batch CPU path; LLM calls with a pool of 8
    def health(self) -> dict                 # versions, bundle sha, policy version and error, llm enabled, breaker state
```

`trace` holds what the logs need and never the customer's text: flags, gate changes, CPU intent/action/confidence,
LLM status/latency/cost, final decision, route reasons, policy version, timings.

### `triage/service.py`

FastAPI app built by `create_app(triage=None)`; module-level `app` for uvicorn. `GET /healthz` returns 503
until the model is loaded and 200 after, with `Triage.health()`; if loading fails it returns 503 with the
error type and message, and the module-level `app` exits with status 3. `POST /triage` as the contract says.
`GET /metrics` returns JSON counters (requests, by status, errors, abstentions, routed, flags, LLM calls by
status, LLM cost, latency p50/p95/p99 over the last 10,000 requests). The latency is measured inside the app, so
it leaves out time queued in uvicorn before the handler runs. Malformed JSON or a bad envelope gets 400 with
`{"error": {"code", "message"}}`; every other valid envelope gets 200 with a schema-valid body, and the response
echoes any id (the body is ASCII JSON, so even a lone surrogate comes back as `\ud800`). Request id from
`X-Request-ID` or generated; echoed in the response header. One JSON log line per request to stdout, no customer
text. An idempotency cache (LRU, 10,000 entries) keyed by the policy in force and the SHA-256 of the canonical
request body returns the same response for a repeated body without a second LLM call. A policy reload (a new
Policy object, with or without a new version) empties it, and an answer computed under the old policy is not
cached. Answers whose LLM step failed (`timeout`, `http_error`, `invalid`, `breaker_open`, `busy`) stay cached
30 s only, so a retry after the outage gets the LLM's answer.

Limits (all fit the published pack, whose largest row is 7,105 bytes):

| Limit | Value | Past it |
|---|---|---|
| Request body | 1 MiB | 413 `body_too_large` |
| Time to receive the body | 10 s | 408 `body_timeout`, connection closed |
| JSON nesting | 64 levels | 400 `bad_json` |
| Bodies over 16 KB | parsed on a 2-thread pool, off the event loop | |
| `POST /triage` requests in the app at once | 128 | 503 `overloaded`, `Retry-After: 1`, counted in /metrics |
| `handle()` calls at once | 16 worker threads | queued |
| LLM call | 4 s hard deadline (`TRIAGE_LLM_TIMEOUT_S`), no retries; 8 waiting at once | CPU answer kept |
| Open connections (uvicorn) | 256 | plain-text 503 from uvicorn, not seen by the app |
| Idle keep-alive (uvicorn) | 5 s | connection closed |

Request time: CPU path about 5 ms in the app (p95 38 ms at 8 clients over HTTP); LLM path the provider's time,
at most 4 s, plus the CPU path. uvicorn has no timeout for a client that opens a connection and sends nothing or
only part of its headers, so 256 such connections block new ones; in production put a proxy with header and
body timeouts and a per-client connection cap in front of the service.

Run: `uvicorn triage.service:app --host 0.0.0.0 --port $PORT --workers 1 --limit-concurrency 256 --timeout-keep-alive 5 --no-access-log`
(`--no-access-log` keeps stdout to the service's JSON lines; `scripts/reproduce.sh serve` runs exactly this).

## Files the evaluation produces

`results/dev_predictions.jsonl`, `results/noisy_predictions.jsonl`, `results/hostile_results.jsonl`,
`results/predictions.jsonl` (all 2,309 rows), `results/latency_bench.json`, all produced through the running
service over HTTP by `scripts/reproduce.sh http-eval`; every run's raw outputs under `results/raw/<run-id>/`;
`results/manifest.json`, rebuilt by `eval` and `http-eval` (`python -m harness.manifest`).
