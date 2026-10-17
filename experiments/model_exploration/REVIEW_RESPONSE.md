# Corrections before confirmation

The Astra design and implementation reviews identified real checks to fix before interpreting new
model results. These changes did not use confirmation outcomes or require retraining the reference.

- Keep the neural action head's internal predicted-intent softmax at temperature 1. Apply exported
  temperatures outside the forward graph. Input projection removes target fields before text/state use.
- Use a new conversation-grouped partition, fitting-only intent/flow mapping and fitting-only
  retrieval. Compare against a reference refit on the same fitting/calibration groups.
- Restore length-bucketed outputs to original row order. Adapt special-token wrapping to the current
  Transformers callable tokenizer API. Real forward/backward benchmark steps then passed.
- Fit correctness on the final prior/mask distribution. Threshold tuning excludes cases that the
  actual shared service path necessarily reviews, and uses its six-decimal confidence rounding.
- Apply adoption rules to exact paired differences and bootstrap quantiles. A synthetic 30/3,030
  gain must not pass a one-point threshold because a displayed number rounds upward.
- Preserve shared gate, observed-action mask, policy review and abstention behavior through a cached
  adapter using `Triage._front/_back`. The optional secondary-intent route remains disabled. Live
  neural serving and HTTP latency still require their own validation before a switch.
- Install frozen review thresholds in the live policy, verify its version, and align classes from
  saved artifacts without reading training labels at service startup. Preserve the original action
  distribution if a mask would remove all probability mass. A pure-classical intent blend can skip
  the unused neural latest-customer probe.
- Score joined development/noisy/hostile output with the official grader before claiming its bars.

The hosted runner received separate cost and interruption checks before bulk calls. Its budget
uses USD/token rates, complete-request byte bounds, provider endpoint price maxima, and completed
call liabilities. A started journal is written before a POST; unresolved attempts remain reserved
and are skipped on resume. Responses are bounded, malformed responses fall back, and finished
predictions use the frozen reference file. Invalid usage metadata also preserves the CPU fallback.
Eligibility uses the uncapped pre-LLM confidence, not the final flag-capped response confidence.

The initial Gemini profile was incompatible because thinking cannot be disabled. The failed calls
and one controlled error diagnostic are retained. Before any selection scores were examined, the
supported profile was set to low reasoning with a 1,024-token output cap and native sampling defaults,
using Google's current documentation. It has its own run ID rather than mixing old and new settings.

Twelve focused offline tests currently cover feature isolation, tokenizer bounds, row ordering,
exact decision arithmetic, complete-request reservations, malformed responses, interruption/resume,
frozen fallbacks, and known versus unknown billing liabilities. They are implementation checks;
they do not establish predictive quality or general production safety.

The real Torch/Transformers head-refinement smoke also passed with a local random tiny BERT,
8 fitting rows and 4 selection rows. It checked cached-versus-direct logits, frozen encoder/intent/
flow weights, finite three-head outputs and checkpoint reload without a Hub request or access to
calibration/confirmation partitions. Run it as `python -m experiments.model_exploration.smoke_head`.
