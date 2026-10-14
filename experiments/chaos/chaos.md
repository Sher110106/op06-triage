# Fault injection: LLM provider faults, 10x load, restarts

**Exploratory, not audited.** One run per scenario (three for 80 clients), 14 Oct 2026, on one Apple M1 laptop
(8 cores, 8 GB) that other work shared. The load average before each run (1.6 to 2.3) is in its `meta.json`.
Every number is on the 600 rows below unless it says otherwise.

The question: would the service survive production? What happens when the LLM provider fails or returns garbage,
at 10x load, and when the process dies or is stopped?

## Setup

- **Service:** `uvicorn triage.service:app --workers 1 --limit-concurrency 256 --timeout-keep-alive 5
  --no-access-log` (the DESIGN.md serve flags), on host 127.0.0.1. It runs the shipped model (`c007c7a3…`) and
  the shipped policy (`2026-10-14.2`). LLM step on: `TRIAGE_LLM=on`,
  `TRIAGE_LLM_BASE_URL=http://127.0.0.1:<port>/v1`, `TRIAGE_LLM_API_KEY=test-key`. The driver removes every
  variable whose name looks like a key, token, secret or proxy from the child processes. No call left the
  machine.
- **Fake provider:** `fake_upstream.py`, an OpenAI-compatible `/chat/completions` server. In `ok` mode it waits a
  fixed 1.5 s (about Luna's p50), then returns the recorded GPT-6 Luna answer for the same conversation
  (`results/raw/llm-tail-luna_V3_none/raw_calls.jsonl`). It finds that answer by the cleaned turns the service puts
  in the prompt; each of the 2,300 dev + dev_noisy rows has its own key. The other modes are in the first table.
- **Rows:** the first 300 dev rows and all 300 dev_noisy rows (600). 148 of them (24.7%) reach the LLM step (CPU
  confidence under 0.642841 and no routing flag), against 20.0% of all 2,300 dev + dev_noisy rows, because half
  this sample is noisy.
- **Load:** `scripts/loadtest.py`, no warm-up, so the breaker starts closed in the measured run. 8 clients unless
  stated. A second client (the "probe") GETs `/healthz` and `/metrics` every 0.25 s and records the breaker
  state and its own round trips.
- **References:** `chaos-cpu-ref` (LLM off, over HTTP) gives each row's CPU-only answer. `chaos-ref-replay` (in
  process, recorded answers replayed) gives what `ok` mode should return.
- **Check per row:** the service logs one line per request with `llm_status`. The driver joins it to the row by
  request id, then compares the answer (id, intent, action, confidence, needs_human) with the CPU-only answer.
- **Accuracy columns:** action accuracy on the rows answered. Read `report.json` "extras" in each run directory.
  `report_official.json` counts the 1,709 rows not sent as wrong, so its numbers mean nothing here.

Reproduce: `.venv/bin/python experiments/chaos/run_chaos.py all` (about 20 minutes), then
`run_chaos.py report` prints the tables below and writes `chaos.json`. Each run is in `results/raw/chaos-<name>/`.

## 1. Provider faults, 8 clients

| Run | Fake provider returns | Errors / non-200 / schema-invalid | p50 / p95 / p99 ms | LLM status (rows) | Breaker opened at | Breaker at end | Rows with a failed LLM call equal to CPU-only | Action acc | Routed |
|---|---|---|---|---|---|---|---|---|---|
| chaos-cpu-ref | (LLM step off) | 0 / 0 / 0 | 34 / 55 / 63 | - | - | - | - | 0.795 | 0.352 |
| chaos-ok | recorded Luna answer after 1.5 s | 0 / 0 / 0 | 14 / 1,545 / 1,557 | ok 133, abstained 15 | - | closed | none failed; 600 of 600 equal the in-process replay (`chaos-ref-replay`) | 0.812 | 0.303 |
| chaos-error500 | HTTP 500 at once | 0 / 0 / 0 | 34 / 55 / 89 | http_error 8, breaker_open 140 | 0.5 s | open | 148 of 148 | 0.795 | 0.352 |
| chaos-down | nothing listens on the port | 0 / 0 / 0 | 34 / 56 / 81 | http_error 7, breaker_open 141 | 0.5 s | open | 148 of 148 | 0.795 | 0.352 |
| chaos-hang | no answer for 30 s | 0 / 0 / 0 | 27 / 45 / 4,042 | timeout 10, breaker_open 138 | 4.4 s | open | 148 of 148 | 0.795 | 0.352 |
| chaos-garbage | HTTP 200, a plain-text sentence | 0 / 0 / 0 | 36 / 55 / 83 | http_error 7, breaker_open 141 | 0.5 s | open | 148 of 148 | 0.795 | 0.352 |
| chaos-garbage_content | HTTP 200, valid envelope, prose content | 0 / 0 / 0 | 35 / 56 / 88 | invalid 8, breaker_open 140 | 0.5 s | open | 148 of 148 | 0.795 | 0.352 |
| chaos-bad_enum | HTTP 200, JSON with action `escalate-to-manager` | 0 / 0 / 0 | 35 / 56 / 77 | invalid 7, breaker_open 141 | 0.5 s | open | 148 of 148 | 0.795 | 0.352 |
| chaos-huge | HTTP 200, a 1 MB body | 0 / 0 / 0 | 35 / 55 / 78 | invalid 8, breaker_open 140 | 0.5 s | open | 148 of 148 | 0.795 | 0.352 |
| chaos-slow | recorded answer after 3.5 s | 0 / 0 / 0 | 16 / 3,549 / 3,560 | ok 133, abstained 15 | - | closed | none failed | 0.812 | 0.303 |
| chaos-flaky | HTTP 500 on half the calls (seeded), else as `ok` | 0 / 0 / 0 | 12 / 1,535 / 1,556 | breaker_open 77, ok 39, http_error 29, abstained 3 | 0.5 s, closed again at 1.8 s | closed | 106 of 106 | 0.802 | 0.337 |
| chaos-constant | HTTP 200, valid JSON, always `offer-refund` | 0 / 0 / 0 | 14 / 1,544 / 1,560 | ok 148 | - | closed | none failed | 0.685 | 0.352 |

Every run ended with SIGTERM and a clean shutdown in 0.14 to 0.26 s.

**What it shows:**

- **No provider fault reached a caller.** In all 11 LLM-on modes, each of the 600 requests got a 200 with a
  schema-valid body. On every row whose LLM call failed (`timeout`, `http_error`, `invalid`, `breaker_open`,
  `busy`), the answer equals the CPU-only answer in all five fields: 1,785 of 1,785 rows across the 21 load runs
  in this file.
- **The breaker opens after 7 to 10 calls, not 5.** Up to 8 calls are already in flight when the fifth failure
  lands, and they still count. Fast failures (500, refused connection, garbage, oversized body) open it within
  0.5 s. After that, LLM-gated rows cost nothing extra: p95 55 to 56 ms against 55 ms with the LLM step off.
- **The worst latency case is a provider that accepts the connection and never answers.** The first 10 LLM rows
  waited the full 4 s deadline (p99 4,042 ms) before the breaker opened at 4.4 s.
- **A slow provider inside the deadline is used, and it sets the tail.** At 3.5 s per call, p95 is 3,549 ms,
  because 24.7% of these rows call the LLM. Nothing below the 4 s deadline limits it.
- **Under a partial outage the breaker opens and closes within seconds.** Failures return at once and successes
  take 1.5 s, so five failures in a row land first and open the breaker at 0.5 s. A success that started before
  the breaker opened then closes it at 1.8 s: `CircuitBreaker.record(ok=True)` closes the breaker from any state,
  so the 30 s open window is not a minimum. In this run the LLM answered 42 of 148 rows (ok or abstained), where
  a provider that fails half its calls could have answered about 74. Action accuracy was 0.802, against 0.812
  when healthy and 0.795 with the step off.
- **Wrong but well-formed answers pass every check.** With the `constant` provider, action accuracy falls from
  0.795 (CPU only) to 0.685. But all 141 changed actions went to a person. A changed action gets confidence
  (CPU confidence + 0) / 2, which is under 0.322 because the step runs only below 0.642841. That is below both
  routing points (0.7104 and 0.8883). So the routed share (0.352) and the accuracy of actions not routed (0.923)
  equal the CPU-only run. The same held in every run: 340 of 340 actions the LLM changed were routed. The cost is
  a wrong suggestion shown to the reviewer on 141 of 600 rows, and no counter shows it: /metrics says `ok`, and
  the breaker stays closed. The share of LLM answers that agree with the CPU action would show it: 48.9% (65 of
  133) in `ok`, 47.4% (203 of 428) over all of dev + dev_noisy in `dev-pipeline-llm-replay`, and 4.7% (7 of 148)
  in `constant`. The service does not log the LLM's action or count agreement.

## 2. Ten times the load, and past the shed point

80 clients is 10x the 8 used for the submitted results. "App" latency is the service's own timer from
`/metrics`; it leaves out time before the handler starts.

| Run | Fake provider | Clients (load-test processes) | 200 / JSON 503 shed / client errors | Client p50 / p95 / p99 ms | App p50 / p95 / p99 ms | Probe `/metrics` p50 / max ms | Load-client CPU / wall s | req/s | LLM status (rows) | Action acc |
|---|---|---|---|---|---|---|---|---|---|---|
| chaos-cpu-c80 | (LLM off) | 80 (1) | 600 / 0 / 0 | 318 / 476 / 493 | 287 / 455 / 470 | 11.8 / 22.8 | 0.84 / 2.7 | 219 | - | 0.795 |
| chaos-ok-c80 | ok | 80 (1) | 600 / 0 / 0 | 319 / 430 / 1,908 | 297 / 421 / 1,894 | 9.0 / 26.5 | 0.75 / 3.4 | 178 | busy 132, ok 13, abstained 3 | 0.798 |
| chaos-ok-c80-r2 | ok | 80 (1) | 600 / 0 / 0 | 324 / 440 / 1,915 | 289 / 432 / 1,906 | 10.5 / 20.7 | 0.76 / 3.4 | 178 | busy 132, ok 13, abstained 3 | 0.798 |
| chaos-ok-c80-r3 | ok | 80 (1) | 600 / 0 / 0 | 312 / 430 / 1,906 | 285 / 421 / 1,884 | 9.4 / 16.2 | 0.76 / 3.4 | 178 | busy 132, ok 13, abstained 3 | 0.797 |
| chaos-slow-c80 | slow (3.5 s) | 80 (1) | 600 / 0 / 0 | 322 / 397 / 3,654 | 293 / 389 / 3,593 | 12.6 / 19.5 | 0.74 / 3.7 | 160 | busy 140, ok 7, abstained 1 | 0.797 |
| chaos-cpu-c128 | (LLM off) | 128 (1) | 600 / 0 / 0 | 497 / 713 / 732 | 464 / 684 / 701 | 13.0 / 19.9 | 1.08 / 2.8 | 216 | - | 0.795 |
| chaos-ok-c200 | ok | 200 (1) | 461 / 139 / 0 | 1,308 / 5,769 / 6,234 | 106 / 1,557 / 1,616 | 4.8 / 17.7 | 6.36 / 10.0 | 60 | busy 74, ok 38, abstained 3 | 0.805 (461 rows) |
| chaos-cpu-c200 | (LLM off) | 200 (1) | 492 / 104 / 4 | 1,337 / 5,691 / 6,320 | 87 / 549 / 612 | 4.7 / 24.4 | 6.08 / 8.1 | 74 | - | 0.795 (492 rows) |
| chaos-ok-c200x4 | ok | 200 (4) | 163 / 437 / 0 | 64 / 613 / 1,685 | 0 / 552 / 1,616 | 11.8 / 92.4 | 1.59 / 2.1 | 285 | busy 33, ok 7, abstained 1 | 0.785 (163 rows) |

**What it shows:**

- **At 80 clients nothing failed.** All 600 requests got a 200 in all five runs, with 0 schema-invalid bodies.
  Client p95 was 397 to 476 ms, and the app's own timer agrees within 35 ms, so the wait is queueing inside the
  service: one thread runs the model (EXPERIMENT_LOG P3-2). Throughput was 178 req/s with the LLM step on and
  219 with it off. The three `ok-c80` repeats agree to within 1 req/s and 10 ms at p95.
- **At 80 clients the LLM step answers few rows.** 132 of 148 LLM-gated rows came back `busy`: at most 8 calls
  may wait on the provider at once, and a ninth keeps the CPU answer. Action accuracy (0.798) is therefore close
  to CPU-only (0.795), not the 0.812 seen at 8 clients. The cap sets a ceiling of 8 calls per provider round
  trip: about 5 calls/s at 1.5 s. That is about 21 req/s of traffic at this sample's 24.7% LLM share, or 27 req/s
  at the 20.0% share of all dev + dev_noisy. Real Luna's p95 is 2.8 s, which lowers the ceiling further.
  `max_in_flight` is not an environment setting, so raising it takes a code change or more processes. The
  `busy` count shows in `/metrics` (`llm.by_status.busy`), not in the response.
- **A slow provider does not slow the CPU rows under load.** With `slow` at 80 clients, p95 is 397 ms, no worse
  than `ok`. Only the 8 rows that got an LLM slot waited 3.5 s (p99 3,654 ms).
- **Up to 128 requests in the app, the service queues; past 128 it sheds.** At 128 clients (LLM off), all 600
  got a 200, with p95 713 ms. Past 128, the service answers at once with a JSON 503 `overloaded` and
  `Retry-After: 1`. With 200 clients from 4 load-test processes, 437 of 600 were shed, 163 answered, 0 errors,
  0 schema-invalid, and client p50 was 64 ms. Most rows were shed because the load test sends its next row at
  once instead of waiting the Retry-After second, so this run measures shedding, not useful throughput.
- **With 200 clients in one load-test process, the client numbers measure the client.** Client p95 was 5.7 to
  5.8 s, but the service's own timer said p50 87 to 106 ms, and the probe got `/metrics` back in 4.7 to 4.8 ms
  (p50). The load-test process used 6.1 to 6.4 s of CPU in 8.1 to 10.0 s of wall time, against 1.08 s in 2.8 s
  at 128 clients. `chaos-cpu-c200` had 4 client errors (`ReadError`). None reached the app (no log line). All 4
  ended within 9 ms of each other, 5.6 s into the run. That fits the server closing connections it saw as idle
  for 5 s (the keep-alive limit) while the overloaded client was about to use them. This cause is not confirmed.
  With 4 load-test processes there were none.

## 3. Provider recovery (`chaos-recovery`)

The provider returned 500 for the first 20 s, then answered well. Traffic was 4 clients at up to 4 requests/s
each, over dev then dev_noisy rows in order, for 75 s.

| Event | Seconds from start |
|---|---|
| Breaker opens (after 7 failed calls) | 1.8 |
| Provider healthy again | 20.1 |
| Breaker half-open; one trial call | 31.7 |
| First request whose LLM answer the service used | 31.8 (11.7 s after the provider recovered) |
| Breaker closed | 33.5 |

All 870 requests got a 200. LLM status: breaker_open 87, ok 66, http_error 7, abstained 1. 58 rows whose LLM call
failed are also in the 600-row CPU reference, and all 58 equal it. With the breaker's 30 s window, the LLM step
returns at most 30 s after the breaker opens plus one call, however early the provider recovers.

## 4. Process faults (`chaos-restart`)

The provider was `ok` unless stated. Traffic was 4 clients at up to 4 requests/s each.

| Test | Result |
|---|---|
| Cold start, LLM step on (model and retrieval index) | port open at 0.57 s; `/healthz` 200 at 3.22 s; first answer 18 ms later |
| `kill -9` under traffic, then a new process on the same port at once (20 ms later) | `/healthz` 200 3.26 s after the kill; first 200 to traffic 3.34 s after the kill |
| Requests in flight at the kill | 3, all lost (`RemoteProtocolError`) |
| Requests during the restart | 11 refused (`ConnectError`) until the port opened 0.56 s after the restart, then 43 JSON 503 `loading` (with `Retry-After: 1`) until the model loaded |
| Answers after the restart | 88 of 88 equal the `chaos-ok` answers for the same rows (80 of 80 before the kill) |
| SIGTERM under traffic | 4 requests in flight, all waiting on the LLM; all 4 got a 200, the last 1.30 s after the signal. The process exited 1.36 s after the signal. 25 requests sent after the signal were refused; 1 sent on an open connection before it closed got a 200 |
| SIGTERM while 8 LLM calls hang (fake provider `hang`, 16 clients) | 16 requests in flight; all 16 got a 200 (8 hit the 4 s deadline and kept the CPU answer). The last came 2.70 s after the signal, and the process exited at 2.79 s |

- **Exit status.** uvicorn 0.54 completes its shutdown ("Application shutdown complete" and "Finished server
  process" are in every log), then raises SIGTERM again. The process therefore ends by signal 15 (status 143 in
  a shell), not 0. A supervisor that counts only 0 as a clean exit will flag every stop.
- **No disk state.** A restart loses only the idempotency cache, the breaker state and the metrics counters.
- **After SIGTERM the port closes.** New requests are refused until a new process is up, so a load balancer must
  stop sending before the signal. Clients should retry on refusal, on `loading` and on `overloaded`. A retried
  request gets the same answer unless the LLM step's outcome differs (for example `busy` the first time).

## Gaps found

1. **No signal for a provider that is up but wrong.** Count LLM/CPU action agreement in `/metrics` and alert when
   it falls far below the 47 to 49% measured here.
2. **The LLM step tops out near 5 calls/s per process** (8 calls in flight, 1.5 s each). Above about 21 to 27
   req/s per process, most LLM-gated rows keep the CPU answer. The cap is not an environment setting.
3. **The breaker flaps under a partial outage**, because any success closes it, including a call that started
   before it opened.
4. **SIGTERM ends with status 143**, from uvicorn, after a full shutdown.
5. **Requests in flight at a crash are lost.** Clients need to retry.

## What this does not show

- One laptop. The Linux box in EXPERIMENT_LOG P3-3 served 84 to 92 req/s on the CPU path, so its load numbers
  would differ.
- The fake provider answers in a fixed time. Real Luna varies (p50 1.5 s, p95 2.8 s per call). There was no
  TLS, DNS or proxy. A 429 with Retry-After was not tested; the code treats any status of 400 or more like a 500
  (`http_error`), which comes from reading the code, not a run.
- Runs last seconds to minutes. Memory growth over hours was not tested.
- One uvicorn worker, as shipped. Several workers would each have their own breaker, cache and LLM cap.
- Clients that send headers slowly were not tested (DESIGN.md: uvicorn has no header timeout; put a proxy in
  front).
- The LLM cost in `/metrics` comes from the fake provider's rough token counts and means nothing here.
