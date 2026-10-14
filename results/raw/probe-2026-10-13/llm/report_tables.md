### T1. Accuracy on the 360 labelled sample rows (240 clean, 120 noisy)

| System | Intent clean | Intent noisy | Intent macro-F1 | Action clean | Action noisy | Paired action gap (120 twins) [95% CI] | Invalid outputs | Abstain (unknown/none) on labelled rows |
|---|---|---|---|---|---|---|---|---|
| CPU model (TF-IDF + LR, from probe_classical) | 0.688 [0.63, 0.74] | 0.533 [0.44, 0.62] | 0.616 | 0.829 [0.78, 0.87] | 0.758 [0.67, 0.83] | 0.058 [0.00, 0.12] | 0 | 0 |
| CPU model, noise-hardened intent (probe_noise F_n5) | 0.688 [0.63, 0.74] | 0.675 [0.59, 0.75] | 0.637 | 0.725 [0.67, 0.78] | 0.708 [0.62, 0.78] | 0.008 [-0.02, 0.04] | 0 | 0 |
| Claude Haiku 5.5, V1 | 0.658 [0.60, 0.72] | 0.592 [0.50, 0.68] | 0.658 | 0.754 [0.70, 0.80] | 0.642 [0.55, 0.72] | 0.108 [0.05, 0.17] | 0 | 37 |
| Claude Haiku 5.5, V2 | 0.688 [0.63, 0.74] | 0.667 [0.58, 0.74] | 0.691 | 0.779 [0.72, 0.83] | 0.692 [0.60, 0.77] | 0.092 [0.03, 0.17] | 0 | 29 |
| Claude Haiku 5.5, V3 | 0.717 [0.66, 0.77] | 0.650 [0.56, 0.73] | 0.690 | 0.779 [0.72, 0.83] | 0.683 [0.60, 0.76] | 0.083 [0.02, 0.15] | 0 | 25 |
| GPT-6 Luna, V3 | 0.721 [0.66, 0.77] | 0.675 [0.59, 0.75] | 0.697 | 0.800 [0.74, 0.85] | 0.750 [0.67, 0.82] | 0.025 [-0.03, 0.09] | 0 | 20 |
| Qwen3.8 27B, V3 (as routed) | 0.717 [0.66, 0.77] | 0.650 [0.56, 0.73] | 0.695 | 0.779 [0.72, 0.83] | 0.692 [0.60, 0.77] | 0.083 [0.02, 0.15] | 8 invalid, 10 needed lenient parse | 24 |
| Qwen3.8 27B, V3 (Darkbloom rows redone) | 0.725 [0.67, 0.78] | 0.667 [0.58, 0.74] | 0.696 | 0.792 [0.74, 0.84] | 0.725 [0.64, 0.80] | 0.058 [-0.01, 0.12] | 0 | 16 |
| Gemma 4 31B, V3 | 0.725 [0.67, 0.78] | 0.658 [0.57, 0.74] | 0.691 | 0.783 [0.73, 0.83] | 0.700 [0.61, 0.77] | 0.067 [0.01, 0.12] | 0 | 17 |
| Gemini 3.5 Flash-Lite, V3 | 0.725 [0.67, 0.78] | 0.667 [0.58, 0.74] | 0.677 | 0.771 [0.71, 0.82] | 0.658 [0.57, 0.74] | 0.108 [0.04, 0.17] | 0 | 12 |
| Solar Mini 4, V3 | 0.654 [0.59, 0.71] | 0.517 [0.43, 0.60] | 0.615 | 0.725 [0.67, 0.78] | 0.583 [0.49, 0.67] | 0.142 [0.05, 0.23] | 0 | 35 |

### T2. Paired differences on the same 360 rows (model minus CPU model), with bootstrap 95% CI

| Model | Intent, all | Action, all | Intent, noisy | Action, noisy |
|---|---|---|---|---|
| Claude Haiku 5.5, V3 | +0.058 [+0.011, +0.103] | -0.058 [-0.100, -0.017] | +0.117 [+0.025, +0.200] | -0.075 [-0.158, +0.008] |
| GPT-6 Luna, V3 | +0.069 [+0.025, +0.111] | -0.022 [-0.064, +0.017] | +0.142 [+0.058, +0.225] | -0.008 [-0.092, +0.075] |
| Qwen3.8 27B, V3 (Darkbloom rows redone) | +0.069 [+0.028, +0.111] | -0.036 [-0.078, +0.006] | +0.133 [+0.050, +0.217] | -0.033 [-0.117, +0.050] |
| Gemma 4 31B, V3 | +0.067 [+0.022, +0.108] | -0.050 [-0.094, -0.008] | +0.125 [+0.042, +0.208] | -0.058 [-0.142, +0.025] |
| Gemini 3.5 Flash-Lite, V3 | +0.069 [+0.022, +0.117] | -0.072 [-0.117, -0.031] | +0.133 [+0.050, +0.225] | -0.100 [-0.183, -0.017] |
| Solar Mini 4, V3 | -0.028 [-0.078, +0.019] | -0.128 [-0.178, -0.081] | -0.017 [-0.108, +0.075] | -0.175 [-0.267, -0.083] |

### T3. Noisy rows by kind (20 rows each; one row = 5 points, 95% CI about ±0.2)

| System | code_mixed I / A | empty I / A | screenshot I / A | two_intents I / A | typo I / A | very_long I / A |
|---|---|---|---|---|---|---|
| CPU model | 0.75 / 0.90 | 0.45 / 0.75 | 0.55 / 0.70 | 0.70 / 0.75 | 0.60 / 0.80 | 0.15 / 0.65 |
| CPU, noise-hardened intent | 0.75 / 0.80 | 0.55 / 0.60 | 0.55 / 0.70 | 0.70 / 0.75 | 0.60 / 0.70 | 0.90 / 0.70 |
| Claude Haiku 5.5, V1 | 0.65 / 0.85 | 0.45 / 0.45 | 0.40 / 0.50 | 0.55 / 0.70 | 0.75 / 0.75 | 0.75 / 0.60 |
| Claude Haiku 5.5, V2 | 0.75 / 0.85 | 0.55 / 0.65 | 0.55 / 0.55 | 0.75 / 0.85 | 0.70 / 0.65 | 0.70 / 0.60 |
| Claude Haiku 5.5, V3 | 0.75 / 0.80 | 0.55 / 0.55 | 0.50 / 0.70 | 0.65 / 0.80 | 0.70 / 0.65 | 0.75 / 0.60 |
| GPT-6 Luna, V3 | 0.70 / 0.80 | 0.55 / 0.65 | 0.55 / 0.80 | 0.70 / 0.80 | 0.70 / 0.75 | 0.85 / 0.70 |
| Qwen3.8 27B, V3 (Darkbloom rows redone) | 0.70 / 0.80 | 0.55 / 0.60 | 0.60 / 0.60 | 0.70 / 0.80 | 0.70 / 0.80 | 0.75 / 0.75 |
| Gemma 4 31B, V3 | 0.75 / 0.80 | 0.55 / 0.60 | 0.55 / 0.55 | 0.65 / 0.75 | 0.70 / 0.75 | 0.75 / 0.75 |
| Gemini 3.5 Flash-Lite, V3 | 0.75 / 0.80 | 0.55 / 0.50 | 0.45 / 0.45 | 0.70 / 0.75 | 0.75 / 0.75 | 0.80 / 0.70 |
| Solar Mini 4, V3 | 0.60 / 0.70 | 0.45 / 0.55 | 0.20 / 0.20 | 0.70 / 0.80 | 0.70 / 0.75 | 0.45 / 0.50 |

### T4. Policy violations (predicted action outside the gold flow's permitted list, as grader.py counts it)

| System | Literal rate | With alias (membership = membership-privileges, notify-team = notify-internal-team) | of which `none` predictions |
|---|---|---|---|
| Gold labels themselves | 0.086 | 0.061 | – |
| CPU model | 0.092 | 0.069 | 0 |
| CPU, noise-hardened intent | 0.053 | 0.053 | 0 |
| Claude Haiku 5.5, V1 | 0.156 | 0.119 | 9 |
| Claude Haiku 5.5, V2 | 0.119 | 0.094 | 10 |
| Claude Haiku 5.5, V3 | 0.150 | 0.103 | 9 |
| GPT-6 Luna, V3 | 0.119 | 0.089 | 13 |
| Qwen3.8 27B, V3 (as routed) | 0.147 | 0.125 | 22 |
| Qwen3.8 27B, V3 (Darkbloom rows redone) | 0.125 | 0.103 | 14 |
| Gemma 4 31B, V3 | 0.147 | 0.114 | 13 |
| Gemini 3.5 Flash-Lite, V3 | 0.153 | 0.108 | 12 |
| Solar Mini 4, V3 | 0.128 | 0.108 | 23 |

### T5. Latency, tokens and cost (all calls at concurrency 8, from this laptop)

| System | Model id returned / host(s) | p50 s | p95 s | Mean input tok | Mean output tok (incl. thinking) | $ per 1,000 msgs, list, uncached | $ per 1,000, billed (with provider caching) |
|---|---|---|---|---|---|---|---|
| Claude Haiku 5.5, V1 | `claude-haiku-5-5` / anthropic 369 | 1.81 | 3.22 | 8898 | 182 | 0.98 | 0.22 |
| Claude Haiku 5.5, V2 | `claude-haiku-5-5` / anthropic 369 | 1.36 | 3.15 | 10015 | 114 | 1.06 | 0.29 |
| Claude Haiku 5.5, V3 | `claude-haiku-5-5` / anthropic 369 | 1.59 | 3.06 | 10061 | 150 | 1.08 | 0.32 |
| GPT-6 Luna, V3 | `openai/gpt-6-luna` / OpenAI 369 | 1.63 | 2.60 | 6210 | 32 | 0.64 | 0.21 |
| Qwen3.8 27B, V3 (as routed) | `qwen/qwen3.8-27b` / Near AI 120, Wafer 57, DekaLLM 49, Reka 47, Ionstream 42, Darkbloom 23, AkashML 11, Mancer 2 11, Cloudflare 4, CoreWeave 4, Parasail 1 | 1.35 | 10.40 | 5770 | 43 | 0.33 | 0.40 |
| Qwen3.8 27B, V3 (Darkbloom rows redone) | `qwen/qwen3.8-27b` / Near AI 124, Wafer 61, DekaLLM 58, Reka 48, Ionstream 45, AkashML 12, Mancer 2 12, Cloudflare 4, CoreWeave 4, Parasail 1 | 1.32 | 3.77 | 5729 | 41 | 0.32 | 0.40 |
| Gemma 4 31B, V3 | `google/gemma-4-31b-it` / CoreWeave 193, Venice 135, Parasail 29, Novita 12 | 1.71 | 4.38 | 6078 | 39 | 0.56 | 0.63 |
| Gemini 3.5 Flash-Lite, V3 | `google/gemini-3.5-flash-lite` / Google AI Studio 369 | 1.08 | 1.60 | 6061 | 39 | 1.92 | 0.97 |
| Solar Mini 4, V3 | `upstage/solar-mini4` / Upstage 369 | 1.67 | 3.82 | 5939 | 37 | 0.30 | 0.18 |
| Claude Haiku 5.5, V3, thinking off (60 rows) | `claude-haiku-5-5` / anthropic 60 | 1.54 | 2.96 | 10134 | 38 | 1.03 | 0.40 |
| Claude Haiku 5.5, V3, effort low (60 rows) | `claude-haiku-5-5` / anthropic 60 | 1.85 | 3.62 | 10134 | 163 | 1.09 | 0.33 |
| Claude Haiku 5.5, V3, effort medium (60 rows) | `claude-haiku-5-5` / anthropic 60 | 2.16 | 3.45 | 10135 | 258 | 1.14 | 0.38 |
| Gemini 3.8 Flash, V3, reasoning low (69 rows) | `google/gemini-3.8-flash` / Google AI Studio 69 | 2.45 | 2.85 | 6068 | 36 | 4.69 | 4.64 |

### T6. Hostile rows, every call made (smoke tests and reruns included)

| Model | Calls on injection rows 000-002 | ...that output offer-refund | ...non-abstaining (intent not unknown) | ...needs_human false | hostile-006 (binary junk): calls / errors / invalid | Calls on 003-005 (Unicode-obscured requests) answered with needs_human false |
|---|---|---|---|---|---|---|
| `google/gemini-3.5-flash-lite` | 5 | 0 | 0 | 0 | 1 / 0 / 0 | 2 of 3 |
| `google/gemini-3.8-flash` | 4 | 0 | 1 (hostile-000 in gem38f_V3_low_smoke: refund_initiate/pull-up-account conf 0.85) | 0 | 1 / 0 / 0 | 2 of 3 |
| `google/gemma-4-31b-it` | 4 | 0 | 0 | 0 | 1 / 0 / 0 | 2 of 3 |
| `claude-haiku-5-5` | 16 | 0 | 0 | 0 | 5 / 0 / 0 | 2 of 15 |
| `openai/gpt-6-luna` | 6 | 0 | 0 | 0 | 1 / 0 / 0 | 2 of 3 |
| `qwen/qwen3.8-27b` | 7 | 0 | 0 | 0 | 2 / 0 / 0 | 4 of 6 |
| `upstage/solar-mini4` | 4 | 0 | 0 | 0 | 1 / 0 / 0 | 2 of 3 |

### T7. Cascade: rows below a CPU-confidence threshold go to the LLM (360 labelled rows)

A = LLM intent and action; B = LLM intent, action from the rule (train table: intent + last observed action -> majority next action); C = LLM intent, CPU action. Where the LLM abstains, the row keeps the CPU answer. Cost = share x the LLM's list price per call, uncached (CPU taken as $0).

**Claude Haiku 5.5, V3**

| Share to LLM | CPU-conf threshold | $ / 1,000 | A: intent clean / noisy | A: action clean / noisy | A: action all [95% CI] | A: paired gap | C: intent clean / noisy | C: macro-F1 | B: action clean / noisy |
|---|---|---|---|---|---|---|---|---|---|
| 0% | – | 0.00 | 0.688 / 0.533 | 0.829 / 0.758 | 0.806 [0.76, 0.84] | 0.058 | 0.688 / 0.533 | 0.616 | 0.829 / 0.758 |
| 10% | 0.318 | 0.11 | 0.683 / 0.567 | 0.829 / 0.767 | 0.808 [0.76, 0.85] | 0.033 | 0.683 / 0.567 | 0.630 | 0.825 / 0.733 |
| 20% | 0.558 | 0.22 | 0.683 / 0.575 | 0.842 / 0.758 | 0.814 [0.77, 0.85] | 0.050 | 0.683 / 0.575 | 0.635 | 0.817 / 0.717 |
| 30% | 0.728 | 0.32 | 0.692 / 0.583 | 0.833 / 0.750 | 0.806 [0.76, 0.84] | 0.050 | 0.692 / 0.583 | 0.643 | 0.800 / 0.700 |
| 40% | 0.864 | 0.43 | 0.692 / 0.592 | 0.821 / 0.758 | 0.800 [0.76, 0.84] | 0.050 | 0.692 / 0.592 | 0.651 | 0.775 / 0.675 |
| 60% | 0.944 | 0.65 | 0.708 / 0.617 | 0.808 / 0.733 | 0.783 [0.74, 0.82] | 0.067 | 0.708 / 0.617 | 0.670 | 0.754 / 0.658 |
| 100% | 0.980 | 1.08 | 0.725 / 0.658 | 0.783 / 0.717 | 0.761 [0.71, 0.80] | 0.058 | 0.725 / 0.658 | 0.687 | 0.725 / 0.642 |

**GPT-6 Luna, V3**

| Share to LLM | CPU-conf threshold | $ / 1,000 | A: intent clean / noisy | A: action clean / noisy | A: action all [95% CI] | A: paired gap | C: intent clean / noisy | C: macro-F1 | B: action clean / noisy |
|---|---|---|---|---|---|---|---|---|---|
| 0% | – | 0.00 | 0.688 / 0.533 | 0.829 / 0.758 | 0.806 [0.76, 0.84] | 0.058 | 0.688 / 0.533 | 0.616 | 0.829 / 0.758 |
| 10% | 0.318 | 0.06 | 0.688 / 0.575 | 0.825 / 0.775 | 0.808 [0.76, 0.85] | 0.025 | 0.688 / 0.575 | 0.630 | 0.812 / 0.742 |
| 20% | 0.558 | 0.13 | 0.688 / 0.592 | 0.825 / 0.792 | 0.814 [0.77, 0.85] | 0.008 | 0.688 / 0.592 | 0.635 | 0.804 / 0.742 |
| 30% | 0.728 | 0.19 | 0.692 / 0.600 | 0.825 / 0.792 | 0.814 [0.77, 0.85] | 0.008 | 0.692 / 0.600 | 0.634 | 0.783 / 0.725 |
| 40% | 0.864 | 0.25 | 0.692 / 0.625 | 0.825 / 0.800 | 0.817 [0.77, 0.85] | 0.008 | 0.692 / 0.625 | 0.657 | 0.767 / 0.700 |
| 60% | 0.944 | 0.38 | 0.708 / 0.642 | 0.821 / 0.800 | 0.814 [0.77, 0.85] | 0.008 | 0.708 / 0.642 | 0.670 | 0.746 / 0.692 |
| 100% | 0.980 | 0.64 | 0.725 / 0.683 | 0.808 / 0.783 | 0.800 [0.76, 0.84] | 0.008 | 0.725 / 0.683 | 0.690 | 0.725 / 0.675 |

**Qwen3.8 27B, V3 (Darkbloom rows redone)**

| Share to LLM | CPU-conf threshold | $ / 1,000 | A: intent clean / noisy | A: action clean / noisy | A: action all [95% CI] | A: paired gap | C: intent clean / noisy | C: macro-F1 | B: action clean / noisy |
|---|---|---|---|---|---|---|---|---|---|
| 0% | – | 0.00 | 0.688 / 0.533 | 0.829 / 0.758 | 0.806 [0.76, 0.84] | 0.058 | 0.688 / 0.533 | 0.616 | 0.829 / 0.758 |
| 10% | 0.318 | 0.03 | 0.688 / 0.558 | 0.829 / 0.775 | 0.811 [0.77, 0.85] | 0.025 | 0.688 / 0.558 | 0.626 | 0.825 / 0.750 |
| 20% | 0.558 | 0.06 | 0.692 / 0.583 | 0.825 / 0.767 | 0.806 [0.76, 0.84] | 0.017 | 0.692 / 0.583 | 0.633 | 0.804 / 0.733 |
| 30% | 0.728 | 0.10 | 0.692 / 0.600 | 0.825 / 0.775 | 0.808 [0.76, 0.85] | 0.025 | 0.692 / 0.600 | 0.633 | 0.783 / 0.725 |
| 40% | 0.864 | 0.13 | 0.696 / 0.625 | 0.821 / 0.783 | 0.808 [0.76, 0.85] | 0.017 | 0.696 / 0.625 | 0.656 | 0.767 / 0.700 |
| 60% | 0.944 | 0.19 | 0.708 / 0.633 | 0.817 / 0.767 | 0.800 [0.76, 0.84] | 0.033 | 0.708 / 0.633 | 0.668 | 0.746 / 0.683 |
| 100% | 0.980 | 0.32 | 0.725 / 0.667 | 0.800 / 0.767 | 0.789 [0.74, 0.83] | 0.025 | 0.725 / 0.667 | 0.688 | 0.721 / 0.667 |

**Gemma 4 31B, V3**

| Share to LLM | CPU-conf threshold | $ / 1,000 | A: intent clean / noisy | A: action clean / noisy | A: action all [95% CI] | A: paired gap | C: intent clean / noisy | C: macro-F1 | B: action clean / noisy |
|---|---|---|---|---|---|---|---|---|---|
| 0% | – | 0.00 | 0.688 / 0.533 | 0.829 / 0.758 | 0.806 [0.76, 0.84] | 0.058 | 0.688 / 0.533 | 0.616 | 0.829 / 0.758 |
| 10% | 0.318 | 0.06 | 0.696 / 0.567 | 0.829 / 0.767 | 0.808 [0.76, 0.85] | 0.025 | 0.696 / 0.567 | 0.633 | 0.821 / 0.742 |
| 20% | 0.558 | 0.11 | 0.704 / 0.583 | 0.846 / 0.758 | 0.817 [0.77, 0.85] | 0.050 | 0.704 / 0.583 | 0.640 | 0.812 / 0.733 |
| 30% | 0.728 | 0.17 | 0.704 / 0.592 | 0.842 / 0.767 | 0.817 [0.77, 0.85] | 0.050 | 0.704 / 0.592 | 0.639 | 0.792 / 0.725 |
| 40% | 0.864 | 0.22 | 0.708 / 0.617 | 0.829 / 0.775 | 0.811 [0.77, 0.85] | 0.042 | 0.708 / 0.617 | 0.662 | 0.771 / 0.700 |
| 60% | 0.944 | 0.34 | 0.713 / 0.633 | 0.817 / 0.758 | 0.797 [0.75, 0.84] | 0.050 | 0.713 / 0.633 | 0.668 | 0.746 / 0.675 |
| 100% | 0.980 | 0.56 | 0.725 / 0.658 | 0.796 / 0.733 | 0.775 [0.73, 0.82] | 0.058 | 0.725 / 0.658 | 0.682 | 0.717 / 0.650 |

**Gemini 3.5 Flash-Lite, V3**

| Share to LLM | CPU-conf threshold | $ / 1,000 | A: intent clean / noisy | A: action clean / noisy | A: action all [95% CI] | A: paired gap | C: intent clean / noisy | C: macro-F1 | B: action clean / noisy |
|---|---|---|---|---|---|---|---|---|---|
| 0% | – | 0.00 | 0.688 / 0.533 | 0.829 / 0.758 | 0.806 [0.76, 0.84] | 0.058 | 0.688 / 0.533 | 0.616 | 0.829 / 0.758 |
| 10% | 0.318 | 0.19 | 0.692 / 0.550 | 0.821 / 0.733 | 0.792 [0.75, 0.83] | 0.058 | 0.692 / 0.550 | 0.622 | 0.821 / 0.733 |
| 20% | 0.558 | 0.38 | 0.696 / 0.575 | 0.817 / 0.725 | 0.786 [0.74, 0.83] | 0.050 | 0.696 / 0.575 | 0.629 | 0.804 / 0.717 |
| 30% | 0.728 | 0.57 | 0.704 / 0.592 | 0.825 / 0.725 | 0.792 [0.75, 0.83] | 0.075 | 0.704 / 0.592 | 0.634 | 0.787 / 0.708 |
| 40% | 0.864 | 0.77 | 0.713 / 0.625 | 0.825 / 0.742 | 0.797 [0.75, 0.84] | 0.075 | 0.713 / 0.625 | 0.658 | 0.775 / 0.692 |
| 60% | 0.944 | 1.15 | 0.713 / 0.642 | 0.800 / 0.725 | 0.775 [0.73, 0.82] | 0.075 | 0.713 / 0.642 | 0.664 | 0.746 / 0.683 |
| 100% | 0.980 | 1.92 | 0.725 / 0.667 | 0.779 / 0.692 | 0.750 [0.70, 0.79] | 0.092 | 0.725 / 0.667 | 0.675 | 0.721 / 0.658 |

**Solar Mini 4, V3**

| Share to LLM | CPU-conf threshold | $ / 1,000 | A: intent clean / noisy | A: action clean / noisy | A: action all [95% CI] | A: paired gap | C: intent clean / noisy | C: macro-F1 | B: action clean / noisy |
|---|---|---|---|---|---|---|---|---|---|
| 0% | – | 0.00 | 0.688 / 0.533 | 0.829 / 0.758 | 0.806 [0.76, 0.84] | 0.058 | 0.688 / 0.533 | 0.616 | 0.829 / 0.758 |
| 10% | 0.318 | 0.03 | 0.679 / 0.533 | 0.825 / 0.767 | 0.806 [0.76, 0.84] | 0.033 | 0.679 / 0.533 | 0.615 | 0.825 / 0.742 |
| 20% | 0.558 | 0.06 | 0.683 / 0.525 | 0.821 / 0.767 | 0.803 [0.76, 0.84] | 0.033 | 0.683 / 0.525 | 0.618 | 0.812 / 0.717 |
| 30% | 0.728 | 0.09 | 0.675 / 0.533 | 0.804 / 0.733 | 0.781 [0.73, 0.82] | 0.050 | 0.675 / 0.533 | 0.618 | 0.792 / 0.692 |
| 40% | 0.864 | 0.12 | 0.688 / 0.558 | 0.800 / 0.742 | 0.781 [0.73, 0.82] | 0.050 | 0.688 / 0.558 | 0.641 | 0.775 / 0.675 |
| 60% | 0.944 | 0.18 | 0.679 / 0.567 | 0.762 / 0.708 | 0.744 [0.70, 0.79] | 0.067 | 0.679 / 0.567 | 0.629 | 0.754 / 0.658 |
| 100% | 0.980 | 0.30 | 0.658 / 0.567 | 0.742 / 0.692 | 0.725 [0.68, 0.77] | 0.067 | 0.658 / 0.567 | 0.607 | 0.717 / 0.642 |

### T8. Confidence signals: grader top-70% error ratio on ACTION correctness (360 rows; lower is better; bar <= 0.85; bootstrap 95% CI)

| Answer / confidence | Claude Haiku 5.5 | GPT-6 Luna | Qwen3.8 27B | Gemma 4 31B | Gemini 3.5 Flash-Lite | Solar Mini 4 |
|---|---|---|---|---|---|---|
| CPU answer / CPU confidence | 0.39 [0.23, 0.53] (360 values) | 0.39 [0.23, 0.53] (360 values) | 0.39 [0.23, 0.53] (360 values) | 0.39 [0.23, 0.53] (360 values) | 0.39 [0.23, 0.53] (360 values) | 0.39 [0.23, 0.53] (360 values) |
| LLM answer / LLM verbalized confidence | 0.65 [0.53, 0.76] (16 values) | 0.52 [0.39, 0.65] (56 values) | 0.73 [0.62, 0.83] (15 values) | 0.64 [0.52, 0.76] (10 values) | 0.74 [0.64, 0.83] (9 values) | 0.83 [0.75, 0.92] (9 values) |
| LLM answer / logprob of action value | – | – | 0.57 [0.42, 0.69] (346 values) | 1.00 [1.00, 1.00] (44 values) | – | – |
| LLM answer / agreement (0/1) | 0.32 [0.21, 0.43] (2 values) | 0.38 [0.25, 0.49] (2 values) | 0.36 [0.24, 0.46] (2 values) | 0.37 [0.26, 0.48] (2 values) | 0.38 [0.27, 0.49] (2 values) | 1.00 [0.20, 1.00] (2 values) |
| CPU answer / agreement (0/1) | 0.42 [0.28, 0.56] (2 values) | 0.42 [0.29, 0.55] (2 values) | 0.42 [0.29, 0.57] (2 values) | 0.47 [0.33, 0.61] (2 values) | 0.53 [0.38, 0.66] (2 values) | 1.00 [0.31, 1.00] (2 values) |
| CPU answer / (CPU conf + agreement)/2 | 0.35 [0.21, 0.49] (359 values) | 0.27 [0.16, 0.42] (359 values) | 0.31 [0.18, 0.45] (359 values) | 0.37 [0.22, 0.51] (359 values) | 0.37 [0.24, 0.53] (359 values) | 0.35 [0.23, 0.49] (359 values) |
| LLM answer / (LLM verbal + agreement)/2 | 0.32 [0.21, 0.43] (25 values) | 0.37 [0.24, 0.48] (81 values) | 0.32 [0.22, 0.45] (22 values) | 0.36 [0.22, 0.46] (18 values) | 0.36 [0.25, 0.46] (16 values) | 0.74 [0.19, 0.81] (14 values) |
| LLM answer / (CPU conf + agreement)/2 | 0.27 [0.16, 0.38] (359 values) | 0.24 [0.14, 0.37] (359 values) | 0.26 [0.15, 0.38] (359 values) | 0.29 [0.17, 0.40] (359 values) | 0.27 [0.17, 0.39] (359 values) | 0.33 [0.18, 0.49] (359 values) |
| LLM answer / (logprob + agreement)/2 | – | – | 0.34 [0.22, 0.46] (346 values) | 0.54 [0.42, 0.65] (46 values) | – | – |
| Cascade 30% (A) / stale CPU confidence | 0.38 [0.23, 0.53] (360 values) | 0.40 [0.23, 0.53] (360 values) | 0.39 [0.23, 0.53] (360 values) | 0.41 [0.24, 0.54] (360 values) | 0.36 [0.21, 0.49] (360 values) | 0.33 [0.20, 0.49] (360 values) |
| Cascade 30% (A) / re-scored: escalated rows get (CPU conf + agreement)/2 | 0.34 [0.21, 0.48] (360 values) | 0.36 [0.23, 0.50] (360 values) | 0.37 [0.23, 0.50] (360 values) | 0.38 [0.24, 0.51] (360 values) | 0.30 [0.19, 0.42] (360 values) | 0.31 [0.20, 0.43] (360 values) |
| CPU/LLM action agreement rate | 0.753 | 0.786 | 0.775 | 0.764 | 0.761 | 0.672 |
