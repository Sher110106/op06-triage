"""Ablations of the shipped v1 pipeline on dev (2,000 clean + 300 noisy + 9 hostile rows).

Each ablation takes out or swaps one part, runs all 2,309 rows in process, writes results/raw/ablation-<name>/
(predictions.jsonl, traces.jsonl.gz, meta.json, the harness report and ablation.json with paired differences), and
compares with its reference: results/raw/dev-pipeline-cpu (LLM off) for CPU ablations,
results/raw/dev-pipeline-llm-replay (recorded GPT-6 Luna answers replayed) for LLM ablations. No API calls.

  python experiments/ablations/run.py                 # every run in order, then the table
  python experiments/ablations/run.py NAME [NAME ...] # some runs
  python experiments/ablations/run.py table           # gather ablation.json files -> ablations.json, print the table
  python experiments/ablations/run.py llm-share-fold-b  # exploratory: the 10/20/30% LLM shares on train fold B
  python experiments/ablations/run.py list
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (HERE, RAW, REF_CPU, REF_LLM, ROOT, SHIPPED_GATE, P, build, cpu_conf_quantile,  # noqa: E402
                    execute)
from train_nostate import OUT as BUNDLES, NoStatePredictor  # noqa: E402

from triage import classical as C  # noqa: E402
from triage import gate as G  # noqa: E402
from triage.policy import DEFAULT_POLICY  # noqa: E402

RUNS: dict[str, tuple[str, str, object]] = {}     # name -> (group, what changed, function)


def ablation(name: str, group: str, changed: str):
    def deco(fn):
        RUNS[name] = (group, changed, fn)
        return fn
    return deco


# ----------------------------------------------------------------------------- controls

@ablation("ref-llm", "control", "nothing: shipped pipeline, recorded Luna answers replayed at the shipped gate "
          "0.642841; must reproduce dev-pipeline-llm-replay")
def ref_llm(changed):
    return execute("ref-llm", build(llm_gate=SHIPPED_GATE), REF_LLM, changed)


@ablation("state-copy", "control", "nothing: CPU model retrained by this folder's copy of the training code with the "
          "state block kept (retraining noise for ablation 6)")
def state_copy(changed):
    t = build(bundle=_bundle("on"))
    t.cpu.__class__ = NoStatePredictor
    return execute("state-copy", t, REF_CPU, changed, extra={"bundle_meta": t.cpu.b.get("meta")})


# ----------------------------------------------------------------------------- CPU ablations (LLM off)

@ablation("no-dedupe", "cpu", "gate keeps repeated sentences (the dedupe_text step removed)")
def no_dedupe(changed):
    with mock.patch.object(G, "_dedupe", lambda s: s):
        return execute("no-dedupe", build(), REF_CPU, changed, slices=("very_long",))


class NoPrior(P.Triage):
    """Every action permitted in every flow, so the prior multiplies by 1: action = argmax of raw probabilities."""

    def _prior(self, policy):
        return np.ones((len(self.cpu.fcls), len(self.cpu.acts)), dtype=np.float32)


@ablation("no-prior", "cpu", "no policy prior: action = argmax of the raw action probabilities (the stacker "
          "and the off-policy check are unchanged)")
def no_prior(changed):
    return execute("no-prior", build(cls=NoPrior), REF_CPU, changed)


def hard_prior(pa, pf, perm):
    """Hard mask: keep only the actions the argmax flow (of the flow model) permits."""
    one = np.zeros_like(pf)
    one[np.arange(len(pf)), pf.argmax(1)] = 1.0
    return C.apply_prior(pa, one, perm)


@ablation("hard-mask", "cpu", "hard mask: only actions permitted by the flow model's argmax flow, instead of the "
          "soft expected permission")
def hard_mask(changed):
    with mock.patch.object(P, "apply_prior", hard_prior):
        return execute("hard-mask", build(), REF_CPU, changed)


def literal_policy(base):
    raw = tomllib.loads(Path(DEFAULT_POLICY).read_text(encoding="utf-8"))
    return dataclasses.replace(base, version=base.version + "+no-alias", alias={},
                               flow_permitted={f: frozenset(a) for f, a in raw["flows"].items()})


@ablation("no-alias", "cpu", "no alias: literal published lists, so 'membership' is not permitted where only "
          "'membership-privileges' is (prior and off-policy check both)")
def no_alias(changed):
    t = build()
    pol = literal_policy(t.store.current())
    t.store.current = lambda: pol
    lost = sorted(f for f in pol.flow_permitted if "membership-privileges" in pol.flow_permitted[f])
    return execute("no-alias", t, REF_CPU, changed, extra={"flows_losing_membership": lost})


@ablation("conf-top-p", "cpu", "confidence = top action probability (after the prior) instead of the stacked "
          "correctness model; shipped thresholds kept")
def conf_top_p(changed):
    t = build()
    t.cpu.confidence = lambda pa, pi, pf, st, ti: pa.max(1)
    return execute("conf-top-p", t, REF_CPU, changed)


def _bundle(state: str) -> Path:
    path = BUNDLES[state]
    if not path.exists():
        subprocess.run([sys.executable, str(HERE / "train_nostate.py"), "--state", state], check=True)
    return path


@ablation("no-state", "cpu", "action model retrained on fold A without the dialogue-state block (text views "
          "only); stacker refit on fold B as in training")
def no_state(changed):
    t = build(bundle=_bundle("off"))
    t.cpu.__class__ = NoStatePredictor
    also = {"state_copy": RAW / "ablation-state-copy"} if (RAW / "ablation-state-copy/predictions.jsonl").exists() else {}
    return execute("no-state", t, REF_CPU, changed, extra={"bundle_meta": t.cpu.b.get("meta")}, compare_also=also)


# ----------------------------------------------------------------------------- LLM step

@ablation("no-llm", "llm", "LLM step off (the CPU pipeline; must equal dev-pipeline-cpu)")
def no_llm(changed):
    res = execute("no-llm", build(), REF_LLM, changed)
    same = (RAW / "ablation-no-llm/predictions.jsonl").read_bytes() == (REF_CPU / "predictions.jsonl").read_bytes()
    _annotate("no-llm", identical_to_dev_pipeline_cpu=same)
    res["identical_to_dev_pipeline_cpu"] = same
    return res


def _llm_share(name, q, changed):
    gate = cpu_conf_quantile(q)
    return execute(name, build(llm_gate=gate), REF_LLM, changed,
                   extra={"llm_share_target": q, "llm_gate_unrounded": gate})


@ablation("llm-10", "llm", "LLM step on the 10% least-confident share: gate = 10% quantile of dev CPU "
          "confidence (0.433508) instead of the 20% one (0.642841)")
def llm_10(changed):
    return _llm_share("llm-10", 0.10, changed)


@ablation("llm-30", "llm", "LLM step on the 30% least-confident share: gate = 30% quantile of dev CPU "
          "confidence (0.795136) instead of the 20% one (0.642841)")
def llm_30(changed):
    return _llm_share("llm-30", 0.30, changed)


# ----------------------------------------------------------------------------- gate

@ablation("gate-noroute", "gate", "gate cleans but its ROUTE flags are ignored (no routing, no cap, LLM not "
          "blocked); abstention for unreadable input kept")
def gate_noroute(changed):
    with mock.patch.object(G, "ROUTE_FLAGS", frozenset()):
        return execute("gate-noroute", build(), REF_CPU, changed)


def raw_clean(obj):
    """No gate: turns passed as given (last 8, unknown speaker -> customer, non-string text -> ""), no cleaning,
    no flags, no abstention. The model's own clean_text (NFKC, zero-width and control removal, 2,000-char cap)
    still runs inside the feature code."""
    obj = obj if isinstance(obj, dict) else {}
    raw = obj.get("context")
    turns = [t for t in raw if isinstance(t, dict)][-G.MAX_TURNS:] if isinstance(raw, list) else []
    ctx = [{"speaker": t.get("speaker") if t.get("speaker") in G.SPEAKERS else "customer",
            "text": t.get("text") if isinstance(t.get("text"), str) else ""} for t in turns]
    ti = obj.get("turn_index")
    ti = ti if isinstance(ti, int) and not isinstance(ti, bool) and 0 <= ti < G.TURN_INDEX_MAX else len(ctx)
    return G.GateResult(obj.get("id") if isinstance(obj.get("id"), str) else "", ctx, ti, frozenset(), None)


@ablation("gate-off", "gate", "no gate at all: no cleaning, no flags, no routing by flags, no abstention "
          "(envelope check kept)")
def gate_off(changed):
    with mock.patch.object(G, "clean", raw_clean), mock.patch.object(G, "ROUTE_FLAGS", frozenset()):
        return execute("gate-off", build(), REF_CPU, changed, slices=("very_long",))


@ablation("gate-no-nodialogue", "gate", "exploratory: gate on, but the no_dialogue flag does not route (the "
          "content flags injection, controls, bidi, surrogates, junk, no_known_text still do)")
def gate_no_nodialogue(changed):
    with mock.patch.object(G, "ROUTE_FLAGS", G.ROUTE_FLAGS - {"no_dialogue"}):
        return execute("gate-no-nodialogue", build(), REF_CPU, changed)


@ablation("gate-off-nopol", "gate", "exploratory: no gate and off-policy routing off too (only the confidence "
          "thresholds route; close to v0 with the shipped thresholds)")
def gate_off_nopol(changed):
    with mock.patch.object(G, "clean", raw_clean), mock.patch.object(G, "ROUTE_FLAGS", frozenset()):
        return execute("gate-off-nopol", build(settings=P.Settings(route_off_policy=False)), REF_CPU, changed)


# ----------------------------------------------------------------------------- LLM share on fold B (exploratory)

def llm_share_fold_b():
    """The dev gates for the 10% / 20% / 30% shares applied on train fold B (3,966 rows no model choice saw):
    out-of-fold CPU confidence, recorded Luna answers for fold B, paired against the LLM step off. Not graded by
    the official grader (fold B is train data); written to experiments/ablations/llm_share_fold_b.json."""
    from common import Replay
    from harness.eval import paired
    sys.path.insert(0, str(ROOT / "experiments/routing"))
    from analyse_routing import LLM_B, OOF_B, OOFTriage
    from triage.data import project
    from triage.llm_action import LLMConfig
    oof = {d["id"]: d["confidence"] for d in map(json.loads, OOF_B.read_text().splitlines())}
    _, B = C.folds()
    rows = [dict(r, _group="clean") for r in B]
    reqs = [project(r) for r in B]

    def run(gate):
        t = OOFTriage(oof, llm_cfg=LLMConfig(mode="off"))
        if gate is not None:
            t.llm = Replay(LLM_B, gate)
        out = t.handle_many(reqs)
        preds = {r["id"]: resp for r, (resp, _) in zip(B, out)}
        return preds, sum(tr.get("llm") is not None for _, tr in out)

    keys = ("action_acc", "top70_error_ratio")
    base, _ = run(None)
    shipped, _ = run(SHIPPED_GATE)
    res = {"rows": len(B), "about": llm_share_fold_b.__doc__, "shares": {}}
    for label, q in (("10%", 0.10), ("20% (shipped)", None), ("30%", 0.30)):
        gate = SHIPPED_GATE if q is None else cpu_conf_quantile(q)
        preds, calls = run(gate)
        res["shares"][label] = {"gate": gate, "llm_calls": calls, "call_share": round(calls / len(B), 4),
                                "vs_llm_off": paired(base, preds, rows=rows, keys=keys)["metrics"],
                                "vs_shipped_20%": paired(shipped, preds, rows=rows, keys=keys)["metrics"]}
    (HERE / "llm_share_fold_b.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v["call_share"], v["vs_llm_off"]["action_acc"]["diff"], v["vs_llm_off"]["action_acc"]["ci95"])
                      for k, v in res["shares"].items()}))
    return res


# ----------------------------------------------------------------------------- table

ORDER = ["ref-llm", "no-llm", "llm-10", "llm-30", "no-dedupe", "no-prior", "hard-mask", "no-alias", "conf-top-p",
         "state-copy", "no-state", "gate-noroute", "gate-off", "gate-no-nodialogue", "gate-off-nopol"]


def _annotate(name, **kv):
    for f in ("meta.json", "ablation.json"):
        p = RAW / f"ablation-{name}" / f
        if p.exists():
            d = json.loads(p.read_text())
            d.update(kv)
            p.write_text(json.dumps(d, indent=1))


def _d(m, k, changed_labels=None, nd=3):
    """'+0.012 [+0.004, +0.020]'; '0 (same labels)' when no label of that kind changed on any labelled row."""
    x = m.get(k)
    if not x:
        return ""
    if x["diff"] == 0 and x["ci95"] == [0.0, 0.0]:
        return "0 (same labels)" if changed_labels == 0 else "0"
    f = lambda v: f"{v:+.{nd}f}".replace("-", "−")
    return f"{f(x['diff'])} [{f(x['ci95'][0])}, {f(x['ci95'][1])}]"


def table():
    out = []
    for name in ORDER:
        p = RAW / f"ablation-{name}" / "ablation.json"
        if p.exists():
            d = json.loads(p.read_text())
            d["name"], d["group"] = name, RUNS[name][0]
            out.append(d)
    (HERE / "ablations.json").write_text(json.dumps({
        "about": "Ablations of the v1 pipeline on dev + dev_noisy (2,300 labelled rows) and the 9 hostile rows. "
                 "Official grader numbers; paired differences against the reference with 95% conversation-bootstrap "
                 "intervals (2,000 draws, harness.eval.paired). CPU ablations: LLM step off, reference "
                 "results/raw/dev-pipeline-cpu. LLM ablations: recorded GPT-6 Luna answers replayed, reference "
                 "results/raw/dev-pipeline-llm-replay. Script: experiments/ablations/run.py.",
        "runs": out,
        "llm_share_fold_b": (json.loads((HERE / "llm_share_fold_b.json").read_text())
                             if (HERE / "llm_share_fold_b.json").exists() else None)}, indent=1))
    head = ("| Run | Ref | Intent | Macro-F1 | Action | Noisy gap | Top-70 | Violation lit. (alias) | Needs human "
            "| Hostile unrouted | Δ intent [95% CI] | Δ macro-F1 | Δ action | Δ top-70 |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for d in out:
        o, m = d["official"], d["paired_vs_reference"]["metrics"]
        n = d["labelled_rows_differing_from_reference"]
        ref = "LLM" if d["reference"].endswith("llm-replay") else "CPU"
        lines.append(f"| {d['name']} | {ref} | {o['intent_acc']:.4f} | {o['macro_f1']:.4f} | {o['action_acc']:.4f} | "
                     f"{o['noisy_gap']:.4f} | {o['top70_error_ratio']:.3f} | {d['policy_violation_literal']:.4f} "
                     f"({d['policy_violation_alias']:.4f}) | {d['needs_human_rate']:.3f} | "
                     f"{d['hostile']['needs_human_false']} / 9 | {_d(m, 'intent_acc', n['intent'])} | "
                     f"{_d(m, 'macro_f1', n['intent'])} | {_d(m, 'action_acc', n['action'])} | "
                     f"{_d(m, 'top70_error_ratio')} |")
    md = "\n".join(lines)
    print(md)
    return md


def main(argv):
    if argv == ["list"]:
        for n in ORDER:
            print(f"{n:19s} {RUNS[n][0]:8s} {RUNS[n][1]}")
        return
    if argv == ["table"]:
        table()
        return
    if argv == ["llm-share-fold-b"]:
        llm_share_fold_b()
        return
    names = argv or ORDER
    bad = [n for n in names if n not in RUNS]
    if bad:
        raise SystemExit(f"unknown run {bad[0]!r}; try: {', '.join(ORDER)}")
    for n in names:
        RUNS[n][2](RUNS[n][1])
    if not argv:
        llm_share_fold_b()
        table()


if __name__ == "__main__":
    main(sys.argv[1:])
