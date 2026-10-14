# Runbook

For whoever runs the triage service. "Normal" figures come from the dev runs named beside them; real traffic will
set its own. `docs/DESIGN.md` lists every setting and limit.

## Start, stop, restart

- **Start:** `scripts/reproduce.sh serve` (`PORT` 8000, `HOST` 0.0.0.0 by default). It checks the settings and the
  policy file first, and builds the retrieval index if the LLM step is on and the index is missing (about 15 s).
  A supervisor should run the uvicorn line from `docs/DESIGN.md`. Exit status 3 means the model, policy or a
  setting failed to load: fix it rather than restart in a loop.
- **Ready** when `GET /healthz` returns 200, 3.2 s after start on the M1 with the LLM step on. Until then
  `POST /triage` gets a JSON 503 `loading` with `Retry-After: 1`.
- **Stop:** SIGTERM. Requests in flight finish (with 8 LLM calls hanging, all 16 got a 200 within 2.7 s); new
  connections are refused at once, so stop sending traffic at the proxy first. A clean stop still exits with
  status 143; under systemd, set `SuccessExitStatus=143` so it is not counted as a failure.
- **Restart:** nothing is on disk; the answer cache, breaker state and counters are lost. After `kill -9` and an
  immediate restart, `/healthz` was 200 in 3.3 s and the 3 requests in flight were lost
  (`experiments/chaos/chaos.md`, section 4). Clients should retry on a refused connection, `loading` or
  `overloaded`.

## Health: `GET /healthz`

Fields: `status`, `bundle`, `bundle_sha256` (shipped `c007c7a3…`), `policy` (`version`, `path`, `last_error`,
`loaded_at`), `llm` (`enabled`, `mode`, `model`, `base_url_host`, `dialect` and `dialect_setting`, `gate`,
`breaker` closed/open/half_open, `breaker_open`, `error`), `started_at`, `uptime_s`. After any change, check
`policy.version`, `llm.enabled` and `llm.dialect`, and that `policy.last_error` and `llm.error` (set when the
retrieval index fails to load) are null.

## Metrics: `GET /metrics`

Counters since start: `requests`, `by_status`, `errors`, `routed`, `abstentions`, `replays`, `flags`,
`route_reasons`, `llm` (`calls`, `by_status`, `cost_usd` at cached list price, `cost_uncached_usd`, token sums),
`latency_ms` (p50/p95/p99 of the last 10,000 requests, timed inside the app, so uvicorn queueing is left out),
`idempotency` and `policy_version`. Normal over 2,309 dev, dev_noisy and hostile requests (each run's
`service_log.jsonl` or `details.jsonl`). The routed shares here are of all 2,309 requests; elsewhere this file and
the README give them over the 2,300 labelled rows (24.3% and 28.2%), which leave out the 9 hostile rows, all routed:

| | LLM on (`dev-http-llm`) | LLM off (`dev-pipeline-cpu`) |
|---|---|---|
| Non-200 answers | 0 | 0 |
| Routed, of 2,309 requests | 567 (24.6%) | 658 (28.5%) |
| Top route reasons | low_confidence 522, off_policy 99 | low_confidence 616, off_policy 87 |
| Abstentions | 2 (the empty and emoji hostile rows) | 2 |
| LLM calls | 460: ok 418, abstained 37, timeout 5 | 0 |
| LLM call time p50 / p95 | 1.64 s / 3.12 s | |
| Request p95, 8 clients, M1 | 1,913 ms | 38 ms (`dev-http-cpu`) |
| LLM cost per message, uncached | $0.000125 | $0 |

`MEMO.md`'s Monday checks come from these fields, as shares of a day's change in each counter (save `/metrics`
each morning; a restart resets it): sent to a person = `routed` ÷ `requests`; unreadable = `abstentions` ÷
`requests`; AI timeouts = `llm.by_status.timeout` ÷ `llm.calls`; AI step on = `/healthz` `llm.breaker` closed;
off-procedure = `route_reasons.off_policy` ÷ `requests`; policy = `/healthz` `policy.version` and `last_error`.

## Logs

One JSON line per request on stdout, with no customer text, intent or action: `ts`, `request_id` (the caller's
`X-Request-ID` if sent), `status`, `latency_ms`, `flags`, `route_reasons`, `needs_human`, `abstained`, `replay`,
`error`, `error_type`, `policy_version`, and the LLM call's status, latency, costs and token counts. A request that
called the LLM also logs what the provider said about its answer: `llm_model`, `llm_provider` and `llm_response_id`
(the completion's id, such as OpenRouter's `gen-...`), each cut to 200 characters; they are null on a replay or
when no call was made. All 6,276 Luna calls in `results/raw/llm-ledger/ledger.jsonl` (3,966 fold-B, 2,300 dev and
10 smoke calls) asked for `openai/gpt-6-luna-20260922` and got back `openai/gpt-6-luna`; another `llm_model` value
means the provider changed what serves the calls. The two live runs predate these fields. Start-up writes a
`loaded` or `load_failed` line; uvicorn writes to stderr. The API key never appears.

## Configuration and the LLM switch

Settings are read at start-up, so a change needs a restart: `TRIAGE_LLM` (on/off/auto), `TRIAGE_LLM_BASE_URL`,
`TRIAGE_LLM_API_KEY`, `TRIAGE_LLM_MODEL`, `TRIAGE_LLM_DIALECT` (auto; openrouter or openai: the request's field
names), `TRIAGE_LLM_PROVIDER` (openai; the OpenRouter host pin, sent only in the openrouter dialect),
`TRIAGE_LLM_TIMEOUT_S` (4.0), `TRIAGE_LLM_GATE` (0.642841), `TRIAGE_POLICY_PATH`, `TRIAGE_ROUTE_OFF_POLICY` (1).
The LLM cap (8 calls waiting) and the breaker (5 failures, 30 s) are fixed in code.

**Two safety properties hold only for the current numbers, and nothing checks them.** A changed action gets
confidence under gate / 2 (now 0.3214), and an agreeing one under (gate + 1) / 2 (now 0.8214). So every action
the LLM changes goes to a person only while both routing points stay above gate / 2, and an LLM answer cannot send
a high-risk action out alone only while the high-risk point (0.8883) stays above (gate + 1) / 2. A gate above
0.7766, or a high-risk point below 0.8214, breaks the second. Check both before changing either.

**To turn the LLM step off**, set `TRIAGE_LLM=off` and restart. On dev, action accuracy falls from 0.8365 to
0.8117, the routed share of labelled rows rises from 24.3% to 28.2%, p95 drops from 1.9 s to 38 ms and LLM cost
goes to zero; intent and hostile handling do not change. To call it on fewer rows, lower `TRIAGE_LLM_GATE`: at a
10% share (0.4335) dev action was 0.8248, against 0.8361 at 20% (recorded answers,
`experiments/ablations/ablations.md`).

## Changing the policy safely

1. Copy `policy/policy.toml`, make one change and raise `version`.
2. Run `.venv/bin/python scripts/replay_policy.py --policy NEW.toml`. It reruns dev and dev_noisy under both files,
   LLM step off, and counts changed actions, routes and violations; exit status 2 means the file has a mistake. Prefer
   `effect = "route"` for a rule that needs an earlier step, since the service sees only 8 turns: with the example
   refund rule at the end of the file, `route` changed no action and routed 8 more rows, while `mask` changed 39
   actions, 26 from right to wrong (`results/raw/policy-replay-refund-{route,mask}/report.txt`; rerun with
   `.venv/bin/python experiments/policy_replay/run.py`).
3. Move the file over `policy/policy.toml` in one step (`mv`). The service looks for a change at most every 2 s,
   on the next request or `/healthz` call, and clears its answer cache when the policy changes.
4. Check `/healthz` for the new version and a null `last_error`, then compare `route_reasons` with the replay.
5. **Roll back** by moving the previous file back the same way.

After a deliberate change to `[flows]`, `scripts/reproduce.sh train` will report that the flows differ from the
published lists; that is expected.

## Playbooks

### LLM provider down, slow or returning garbage

What the fault runs showed (`experiments/chaos/chaos.md`; 600 rows, fake provider, M1):
- **Down, HTTP 500, plain text, unknown actions, 1 MB replies:** no caller saw an error. The breaker opened within
  0.5 s, after 7 or 8 failed calls, and every affected row got its CPU-only answer (148 of 148 in each run; 1,785
  of 1,785 failed-call rows across all 21 fault and load runs).
- **No answer:** the first 10 LLM rows waited the full 4 s before the breaker opened.
- **Slow, under 4 s:** the breaker stays closed and the provider sets the tail (3.5 s calls, p95 3.5 s).
- **Recovery:** one trial call 30 s after the breaker opens; LLM answers returned 11.7 s after the provider did.

Signs: `llm.breaker` not closed; failures or `breaker_open` in `llm.by_status`; routed share moving toward the
LLM-off level. The service stays up at LLM-off accuracy. If the provider stays slow, lower `TRIAGE_LLM_TIMEOUT_S`
or turn the step off. If every call fails with `http_error` from the first request, check `/healthz`
`llm.dialect`: a gateway that rejects unknown fields fails every `openrouter`-dialect call (against the strict
fake, five 400s opened the breaker). A 429 counts as an `http_error`: no retry, and `Retry-After` is not read.

**Up but wrong:** no check catches well-formed wrong answers. When the fake provider always said `offer-refund`,
every changed action went to a person, so the routed share rose to the LLM-off level while `llm.by_status` showed
only `ok`. That pattern is one sign, and a weak one: on dev the two levels are 24.3% and 28.2% of labelled rows.
The other is a new `llm_model` value in the log, which shows only if the provider renames what it serves. There
is no agreement counter; Luna agreed with the CPU action on 47.4% of dev calls in the replay and 48.9% in the
`ok` fault run (`chaos.md`).

### Bad policy file

A saved file with a mistake, or a deleted file, does not take effect: the service keeps the last good policy and
shows the error in `policy.last_error`. Fix and save again. At start-up there is no last good policy: under the
uvicorn line a bad file stops the service with status 3; `scripts/reproduce.sh serve` refuses to start, prints
the error and exits with status 1.

### Overload

Signs: 503s with `errors.overloaded`, a rising `latency_ms.p95` or `llm.by_status.busy`. Up to 128 requests in the
app it queues (128 clients on the M1, LLM step off, 600 rows, `chaos-cpu-c128`: all 200, p95 713 ms); past that it
sheds at once with a JSON 503 and `Retry-After: 1`. Past 256 open connections uvicorn sends a plain-text 503 that
`/metrics` never sees. On the Linux box the CPU path served 84 req/s at 8 clients and 92 at 80 (p95 1.14 s; LLM
step off). At 80 clients on the M1 with a fake provider (600 rows, `chaos-ok-c80`), 132 of 148 LLM-gated rows got
`busy` and kept the CPU answer. Make clients honour `Retry-After`, add processes, and put a proxy with header and
body timeouts and per-client connection caps in front. Slow or idle clients were not tested; that uvicorn has no
header timeout comes from reading its code.

### High routed share or many abstentions

Read `route_reasons`. `low_confidence` up: check the breaker, then `policy_version`, then the traffic (new topics).
Text in another language shows as its own reason, `no_known_text`. `off_policy` up: a policy edit, or conversations the procedure no longer
fits. `injection`, `junk`, `controls_removed` or `bidi_override` up: an attack or an upstream encoding fault; read
a sample upstream, since the log holds no text. Abstentions up: upstream is sending empty contexts. `MEMO.md` says
when to send all work to people.

### Memory

Every store is bounded: 10,000 cached answers, 10,000 latency values, 128 requests, 1 MiB per body. EXPERIMENT_LOG
P3-3 estimates about 250 MB per process without the retrieval index (120 MB on disk). Memory over hours was not
tested. If it climbs, save `/metrics` and restart.

## Scaling

All model work in a process runs on one CPU thread. Spreading it over threads on the two-socket Linux box cut
throughput from about 130 to about 40 req/s (`results/raw/linux-threads/scaling.json`), so scale with processes.
One process is enough for 3,000 messages a day (about 0.035 req/s on average): with the LLM step on, its cap of
about 5 calls/s holds up to about 21 to 27 req/s of traffic before most LLM-gated rows fall back (`chaos.md`). Add
processes when peak traffic nears about 20 req/s.
Each process has its own model, cache, breaker, LLM cap (about 5 calls/s) and counters. Under
`uvicorn --workers N` each `/metrics` call reaches one worker, so separate ports behind a proxy are easier to
watch. Neither layout has been tested.
