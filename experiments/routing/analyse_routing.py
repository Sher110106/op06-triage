"""Phase 2: choose the routing operating point (who goes to a person) and check the routing flags.

The real pipeline runs with its LLM step replaced by a replay of the recorded GPT-6 Luna answers, so every
routing rule is exactly the service's, and no API call is made.

- Fold B (3,966 train rows held out from the CPU models) chooses the thresholds. CPU confidence there is
  out-of-fold (experiments/llm_confirm/cpu_B_predictions.jsonl), because the confidence model was fit on fold B.
- Dev + dev_noisy (2,300 rows, never used to fit anything) reports the result, with the official grader.

Cost model: a person reviewing one message costs 1. A wrong automatic action costs R, set per risk tier in the
policy file (high: refunds, order/account changes, purchases, promo codes, passwords; low: everything else).
For each tier the threshold is the confidence cut that minimises routed + R x wrong-and-not-routed on fold B,
counting only rows that no flag already routes. Defaults: R_high = 10, R_low = 3.

  python experiments/routing/analyse_routing.py [--r-high 10] [--r-low 3]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from triage import pipeline as P  # noqa: E402
from triage.classical import folds  # noqa: E402
from triage.data import load_split, project  # noqa: E402
from triage.llm_action import LLMConfig, LLMResult  # noqa: E402

LLM_B = ROOT / "experiments/llm_confirm/runs/luna_B.jsonl"
LLM_DEV = ROOT / "experiments/llm_tail/runs/luna_V3_none.jsonl"
OOF_B = ROOT / "experiments/llm_confirm/cpu_B_predictions.jsonl"
POLICY = ROOT / "policy/policy.toml"


class Replay:
    """Stands in for LLMActionStep: returns the recorded answer for a row id (same model, prompt and examples)."""

    def __init__(self, path, gate):
        self.cfg = LLMConfig(mode="on", api_key="replay", gate=gate)
        self.rec = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            p = r.get("pred") or {}
            ok = r.get("status") == "ok" and r.get("parse_status") in ("ok", "lenient_ok") and p
            if not ok:
                status = "invalid"
            elif p.get("intent") == "unknown" or p.get("action") == "none":
                status = "abstained"
            else:
                status = "ok"
            self.rec[r["id"]] = LLMResult(status, intent=p.get("intent"), action=p.get("action"),
                                          latency_s=r.get("latency_s") or 0.0, cost_usd=r.get("cost_list_actual"),
                                          model=r.get("model_returned"), provider=r.get("provider_returned"))
        self.calls = 0

    def decide(self, row):
        self.calls += 1
        return self.rec.get(row["id"], LLMResult("http_error", error="no recorded answer"))

    def close(self):
        pass


class OOFTriage(P.Triage):
    """The pipeline with CPU confidence replaced by out-of-fold values (fold B only)."""

    def __init__(self, oof: dict, **kw):
        super().__init__(**kw)
        self.oof = oof

    def _cpu(self, live, policy):
        super()._cpu(live, policy)
        for w in live:
            if w.rid in self.oof:
                w.conf = self.oof[w.rid]


def build(policy_path, replay, oof=None):
    kw = dict(policy_path=policy_path, llm_cfg=LLMConfig(mode="off"))
    t = OOFTriage(oof, **kw) if oof is not None else P.Triage(**kw)
    t.llm = replay
    return t


def rows_info(t, rows):
    """Run the pipeline; return per-row facts needed to choose thresholds (before the threshold is applied)."""
    out = t.handle_many([project(r) for r in rows])
    info = []
    for r, (resp, tr) in zip(rows, out):
        cpu, llm = tr.get("cpu") or {}, tr.get("llm") or {}
        conf = cpu.get("confidence", 0.0)
        if llm.get("used"):
            conf = round((conf + (1.0 if llm["action"] == cpu["action"] else 0.0)) / 2, 6)
        gate_reasons = [x for x in tr["route_reasons"] if x not in ("low_confidence", "off_policy", "second_intent")]
        info.append({"id": r["id"], "action": resp["action"], "intent": resp["intent"], "conf": conf,
                     "flags": tr["flags"], "gate_reasons": gate_reasons, "llm_used": bool(llm.get("used")),
                     "llm_called": bool(llm), "ok": resp["action"] == r.get("action"),
                     "intent_ok": resp["intent"] == r.get("intent"), "abstained": resp["intent"] == "unknown"})
    return info


def flagged(x, cfg):
    """Routed by a flag (not by the threshold) under a routing configuration."""
    return bool(x["gate_reasons"]) or x["abstained"] or (cfg["off_policy"] and "off_policy" in x["flags"]) \
        or (cfg["second_intent"] and "second_intent" in x["flags"])


def choose_threshold(items, R):
    """Cut on confidence minimising routed + R * wrong-and-auto, among rows no flag routes."""
    c = np.array([x["conf"] for x in items])
    ok = np.array([x["ok"] for x in items])
    best = (float("inf"), 0.0)
    for t in np.unique(np.concatenate([[0.0], c, [1.01]])):
        routed = c < t
        cost = routed.sum() + R * (~ok & ~routed).sum()
        if cost < best[0] - 1e-9:
            best = (cost, float(t))
    return best[1]


def tier_of(policy, action):
    return policy.risk.get(action, "low")


def summarise(info, policy, thr, cfg, R=None):
    lab = [x for x in info if not x["id"].startswith("hostile")]
    routed = [flagged(x, cfg) or x["conf"] < thr[tier_of(policy, x["action"])] for x in lab]
    auto = [x for x, r in zip(lab, routed) if not r]
    out = {"rows": len(lab), "routed_share": round(float(np.mean(routed)), 4),
           "auto_action_acc": round(float(np.mean([x["ok"] for x in auto])), 4) if auto else None,
           "routed_action_acc": round(float(np.mean([x["ok"] for x, r in zip(lab, routed) if r])), 4) if any(routed) else None,
           "by_tier": {}}
    if R:
        out["cost_per_100"] = round(100 * (sum(routed) + sum(R[tier_of(policy, x["action"])] for x, r in zip(lab, routed)
                                                         if not r and not x["ok"])) / len(lab), 2)
    for tier in sorted(set(thr)):
        sel = [(x, r) for x, r in zip(lab, routed) if tier_of(policy, x["action"]) == tier]
        a = [x for x, r in sel if not r]
        out["by_tier"][tier] = {"rows": len(sel), "routed_share": round(float(np.mean([r for _, r in sel])), 4) if sel else None,
                                "auto_action_acc": round(float(np.mean([x["ok"] for x in a])), 4) if a else None}
    return out


def flag_check(info, name):
    sel = [x for x in info if name in x["flags"] and not x["id"].startswith("hostile")]
    return {"rows": len(sel), "share": round(len(sel) / max(1, len(info)), 4),
            "action_acc": round(float(np.mean([x["ok"] for x in sel])), 4) if sel else None,
            "intent_acc": round(float(np.mean([x["intent_ok"] for x in sel])), 4) if sel else None}


def write_policy(src: Path, dst: Path, thr: dict):
    lines, in_thr = [], False
    for line in src.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("["):
            in_thr = s == "[thresholds]"
        if in_thr and "=" in s and not s.startswith("#"):
            key = s.split("=")[0].strip()
            if key in thr:
                line = f"{key} = {thr[key]}"
        lines.append(line)
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-high", type=float, default=10.0)
    ap.add_argument("--r-low", type=float, default=3.0)
    ap.add_argument("--route-off-policy", choices=["cost", "on", "off"], default="on",
                    help="'on' (default, decision D1: never act outside the written procedure), 'off', or 'cost' "
                         "(pick whichever routing configuration has the lower fold-B cost)")
    a = ap.parse_args()
    from triage.policy import load
    policy = load(POLICY)
    gate = 0.642841
    oof = {json.loads(l)["id"]: json.loads(l)["confidence"] for l in OOF_B.read_text().splitlines()}
    _, B = folds()

    tb = build(POLICY, Replay(LLM_B, gate), oof=oof)
    info_b = rows_info(tb, B)
    R = {"high": a.r_high, "low": a.r_low}

    def thresholds(cfg, Rx):
        t = {}
        for tier in ("high", "low"):
            items = [x for x in info_b if tier_of(policy, x["action"]) == tier and not flagged(x, cfg)]
            t[tier] = round(choose_threshold(items, Rx[tier]), 4)
        return t

    configs = {}
    for op in (True, False):
        for si in (False, True):
            cfg = {"off_policy": op, "second_intent": si}
            t = thresholds(cfg, R)
            configs[f"off_policy={op},second_intent={si}"] = {"cfg": cfg, "thresholds": t,
                                                               "fold_B": summarise(info_b, policy, t, cfg, R)}
    best_name = min(configs, key=lambda k: configs[k]["fold_B"]["cost_per_100"])
    res_cost_best = best_name
    if a.route_off_policy != "cost":                  # D1: off-policy routing is a policy rule, not a cost choice
        want = a.route_off_policy == "on"
        best_name = min((k for k in configs if configs[k]["cfg"]["off_policy"] == want),
                        key=lambda k: configs[k]["fold_B"]["cost_per_100"])
    cfg, thr = configs[best_name]["cfg"], configs[best_name]["thresholds"]
    sweep = {}
    for rh, rl in ((3, 2), (5, 2), (5, 3), (10, 3), (10, 5), (20, 5), (20, 10)):
        Rx = {"high": rh, "low": rl}
        t2 = thresholds(cfg, Rx)
        sweep[f"R_high={rh},R_low={rl}"] = {"thresholds": t2, "fold_B": summarise(info_b, policy, t2, cfg, Rx)}

    res = {"cost_ratio": R, "routing_configs_fold_B": configs, "chosen_config": best_name,
           "lowest_cost_config": res_cost_best, "off_policy_mode": a.route_off_policy,
           "thresholds_from_fold_B": thr, "llm_gate": gate,
           "fold_B": {"with_chosen_thresholds": summarise(info_b, policy, thr, cfg, R),
                      "v0_threshold_0.8498": summarise(info_b, policy, {"high": 0.8498, "low": 0.8498}, cfg, R),
                      "llm_calls": tb.llm.calls,
                      "flags": {f: flag_check(info_b, f) for f in ("off_policy", "second_intent", "no_dialogue")},
                      "all_rows_action_acc": round(float(np.mean([x["ok"] for x in info_b])), 4)},
           "sweep_fold_B": sweep}

    # dev report with the chosen thresholds, real stacker confidence, recorded Luna answers on dev
    with tempfile.TemporaryDirectory() as d:
        pol = Path(d) / "policy.toml"
        write_policy(POLICY, pol, thr)
        td = build(pol, Replay(LLM_DEV, gate))
        td.settings = P.Settings(route_off_policy=cfg["off_policy"], route_second_intent=cfg["second_intent"])
        dev = load_split("dev") + load_split("noisy") + load_split("hostile")
        out = td.handle_many([project(r) for r in dev])
        run = ROOT / "results/raw/dev-pipeline-llm-replay"
        run.mkdir(parents=True, exist_ok=True)
        with open(run / "predictions.jsonl", "w", encoding="utf-8") as fh:
            for resp, _ in out:
                fh.write(json.dumps(resp, ensure_ascii=False) + "\n")
        with open(run / "traces.jsonl", "w", encoding="utf-8") as fh:
            for _, tr in out:
                fh.write(json.dumps(tr, ensure_ascii=False) + "\n")
        (run / "meta.json").write_text(json.dumps({
            "run_id": run.name, "system": "pipeline with recorded GPT-6 Luna answers replayed (no API calls)",
            "llm_answers": str(LLM_DEV.relative_to(ROOT)), "policy_thresholds": thr, "llm_gate": gate,
            "routing": cfg,
            "llm_rows_called": td.llm.calls}, indent=1))
        subprocess.run([sys.executable, "-m", "harness.eval", "grade", "--pred", str(run / "predictions.jsonl")],
                       cwd=ROOT, check=True)
        rep = json.loads((run / "report.json").read_text())
        tb2 = build(pol, Replay(LLM_DEV, gate))
        tb2.settings = td.settings
        info_d = rows_info(tb2, dev)
    res["dev"] = {"official": rep["official"], "extras": {k: rep["extras"][k] for k in
                                                          ("ci95_conversation_bootstrap", "needs_human_rate",
                                                           "action_acc_when_not_routed", "noise_kinds",
                                                           "policy_violation_alias")},
                  "routing": summarise(info_d, policy, thr, cfg, R), "llm_calls": td.llm.calls,
                  "flags": {f: flag_check(info_d, f) for f in ("off_policy", "second_intent", "no_dialogue")},
                  "second_intent_on_two_intents_rows": flag_check([x for x in info_d if x["id"].endswith("-noisy")
                                                                   and x["id"] in {r["id"] for r in load_split("noisy")
                                                                                   if r["noise_kind"] == "two_intents"}],
                                                                  "second_intent")}
    (HERE / "results_routing.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"config": best_name, "configs": {k: (v["thresholds"], v["fold_B"]["cost_per_100"],
                                                           v["fold_B"]["routed_share"]) for k, v in configs.items()},
                      "thresholds": thr, "fold_B": res["fold_B"]["with_chosen_thresholds"],
                      "dev_routing": res["dev"]["routing"], "dev_official": {k: rep["official"][k] for k in
                      ("intent_acc", "macro_f1", "action_acc", "noisy_gap", "top70_error_ratio")}}, indent=1))


if __name__ == "__main__":
    main()
