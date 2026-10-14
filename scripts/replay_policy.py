"""Replay dev + dev_noisy under a changed policy file and report what would change, before the change ships.

  .venv/bin/python scripts/replay_policy.py --policy NEW.toml [--current policy/policy.toml] [--out FILE]

Both passes run the same CPU model with the LLM step off, so every difference comes from the policy file. The
report counts changed actions and routes, policy violations (as published, and with the membership alias), the
official grader's metrics before and after, and up to 10 example ids for each kind of change. Other TRIAGE_*
settings (routing switches, the bundle) are taken from the environment, as the service would. --out writes the
same report as JSON. Exit status 2 when a policy file has a mistake; the message says what to fix.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.eval import ALIAS, G  # noqa: E402
from triage.data import DATA, load_split, project  # noqa: E402
from triage.policy import DEFAULT_POLICY, PolicyError, PolicyStore, load  # noqa: E402

EXAMPLES = 10
METRICS = [("intent_acc", "intent accuracy"), ("macro_f1", "intent macro-F1"), ("action_acc", "action accuracy"),
           ("action_acc_clean", "action accuracy, clean"), ("action_acc_noisy", "action accuracy, noisy"),
           ("noisy_gap", "clean-to-noisy gap"), ("top70_error_ratio", "top-70% error ratio"),
           ("schema_valid_pct", "schema valid %"), ("policy_violation", "policy violation (grader)"),
           ("needs_human_rate", "needs_human rate"), ("action_acc_not_routed", "action accuracy, not routed")]
KINDS = {"wrong_to_right": "action wrong -> right", "right_to_wrong": "action right -> wrong",
         "wrong_to_wrong": "action wrong -> another wrong", "newly_routed": "newly routed",
         "no_longer_routed": "no longer routed", "new_violation": "new violation",
         "violation_cleared": "violation cleared"}


def violates(action: str, permitted: list, alias: bool) -> bool:
    allowed = set(permitted) | ({ALIAS[a] for a in permitted if a in ALIAS} if alias else set())
    return action not in allowed


def score(triage, labelled: list[dict], hostile: list[dict]) -> tuple[dict, dict, dict]:
    """(responses by id, traces by id, metrics) for one pass over dev + noisy + hostile."""
    rows = labelled + hostile
    out = triage.handle_many([project(r) for r in rows], [f"replay-{i}" for i in range(len(rows))])
    resp = {r["id"]: o[0] for r, o in zip(rows, out)}
    trace = {r["id"]: o[1] for r, o in zip(rows, out)}
    n = len(labelled)
    dev = [r for r in labelled if not r["id"].endswith("-noisy")]
    noisy = [r for r in labelled if r["id"].endswith("-noisy")]
    rep = G.grade(dev, noisy, hostile, resp, json.loads((DATA / "schema.json").read_text()))
    scored = [r for r in labelled if r.get("permitted_actions")]
    lit = sum(violates(resp[r["id"]]["action"], r["permitted_actions"], False) for r in scored)
    ali = sum(violates(resp[r["id"]]["action"], r["permitted_actions"], True) for r in scored)
    auto = [resp[r["id"]]["action"] == r["action"] for r in labelled if not resp[r["id"]]["needs_human"]]
    m = {"intent_acc": rep["intent"]["accuracy_all"], "macro_f1": rep["intent"]["macro_f1_all"],
         "action_acc": rep["action"]["accuracy_all"], "action_acc_clean": rep["action"]["accuracy_clean"],
         "action_acc_noisy": rep["action"]["accuracy_noisy"], "noisy_gap": rep["action"]["noisy_gap"],
         "top70_error_ratio": rep["confidence"]["top70_error_ratio"],
         "schema_valid_pct": round(rep["schema"]["valid_rate_all_inputs"] * 100, 2),
         "policy_violation": rep["policy"]["violation_rate"],
         "needs_human_rate": round(sum(resp[r["id"]]["needs_human"] for r in labelled) / n, 4),
         "action_acc_not_routed": round(sum(auto) / len(auto), 4) if auto else None,
         "violations": {"rows_scored": len(scored), "literal": lit, "literal_rate": round(lit / len(scored), 4),
                        "alias": ali, "alias_rate": round(ali / len(scored), 4)},
         "bars_pass": all(b["pass"] for b in rep["bars"].values() if b["pass"] is not None),
         "route_reasons": dict(Counter(x for r in labelled for x in trace[r["id"]]["route_reasons"])),
         "rows_with_masked_actions": sum(bool((trace[r["id"]].get("cpu") or {}).get("masked")) for r in labelled)}
    return resp, trace, m


def compare(labelled: list[dict], before: dict, after: dict) -> tuple[dict, dict]:
    """Counts and example rows for each kind of change, over the labelled rows."""
    ex = {k: [] for k in KINDS}
    counts = Counter()
    for r in labelled:
        b, a = before[r["id"]], after[r["id"]]
        item = {"id": r["id"], "action": [b["action"], a["action"]], "gold_action": r["action"],
                "needs_human": [b["needs_human"], a["needs_human"]], "confidence": [b["confidence"], a["confidence"]]}
        kinds = []
        if a["action"] != b["action"]:
            counts["actions_changed"] += 1
            kinds.append("wrong_to_right" if a["action"] == r["action"] else
                         "right_to_wrong" if b["action"] == r["action"] else "wrong_to_wrong")
        if a["needs_human"] != b["needs_human"]:
            counts["needs_human_changed"] += 1
            kinds.append("newly_routed" if a["needs_human"] else "no_longer_routed")
        if a["confidence"] != b["confidence"]:
            counts["confidence_changed"] += 1
        if r.get("permitted_actions"):
            vb, va = (violates(x["action"], r["permitted_actions"], False) for x in (b, a))
            if vb != va:
                kinds.append("new_violation" if va else "violation_cleared")
        for k in kinds:
            counts[k] += 1
            if len(ex[k]) < EXAMPLES:
                ex[k].append(item)
    return {k: counts[k] for k in ("actions_changed", "needs_human_changed", "confidence_changed", *KINDS)}, ex


def _load(path: Path):
    try:
        return load(path)
    except PolicyError as e:
        raise PolicyError(f"{path}: {e}") from None


def replay(current: Path, new: Path) -> dict:
    pol_cur, pol_new = _load(current), _load(new)                 # a bad file stops before the model loads
    from triage.pipeline import Triage
    triage = Triage.from_env(dict(os.environ, TRIAGE_LLM="off", TRIAGE_POLICY_PATH=str(current)))
    try:
        labelled = load_split("dev") + load_split("noisy")
        hostile = load_split("hostile")
        before, _, m_before = score(triage, labelled, hostile)
        triage.store = PolicyStore(new)
        after, trace, m_after = score(triage, labelled, hostile)
    finally:
        triage.close()
    counts, examples = compare(labelled, before, after)
    rules = Counter(f for r in labelled for f in trace[r["id"]]["flags"] if f.startswith("rule:"))
    return {"rows": len(labelled), "llm": "off",
            "current": {"path": str(current), "version": pol_cur.version, "rules": len(pol_cur.rules)},
            "new": {"path": str(new), "version": pol_new.version, "rules": len(pol_new.rules)},
            "changes": counts, "metrics": {"current": m_before, "new": m_after},
            "rule_flags_new": dict(rules), "examples": examples}


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.4f}"


def _policy_line(p: dict) -> str:
    return f"{p['path']}  (version {p['version']}, {p['rules']} rule{'' if p['rules'] == 1 else 's'})"


def render(rep: dict) -> str:
    c, mb, ma = rep["changes"], rep["metrics"]["current"], rep["metrics"]["new"]
    out = [f"Policy replay over dev + dev_noisy ({rep['rows']:,} rows), LLM step off",
           f"  current  {_policy_line(rep['current'])}", f"  new      {_policy_line(rep['new'])}",
           "", "Changes",
           f"  actions changed      {c['actions_changed']:>5}   (wrong -> right {c['wrong_to_right']}, "
           f"right -> wrong {c['right_to_wrong']}, wrong -> another wrong {c['wrong_to_wrong']})",
           f"  needs_human changed  {c['needs_human_changed']:>5}   (newly routed {c['newly_routed']}, "
           f"no longer routed {c['no_longer_routed']})",
           f"  confidence changed   {c['confidence_changed']:>5}",
           f"  rows where a rule masks an action   {mb['rows_with_masked_actions']} -> "
           f"{ma['rows_with_masked_actions']}"]
    if rep["rule_flags_new"]:
        out.append("  rule flags under the new policy: "
                   + ", ".join(f"{k} {v}" for k, v in sorted(rep["rule_flags_new"].items())))
    vb, va = mb["violations"], ma["violations"]
    out += ["", f"Policy violations ({vb['rows_scored']:,} rows with a published list)"]
    for k, label in (("literal", "as published"), ("alias", "with the alias")):
        out.append(f"  {label:<21}{vb[k]:>5} ({vb[k + '_rate']:.4f})  ->  {va[k]:>5} ({va[k + '_rate']:.4f})")
    out += ["", f"  {'Metric (official grader unless noted)':<38}{'current':>9}{'new':>9}{'change':>9}"]
    for k, label in METRICS:
        b, a = mb[k], ma[k]
        d = f"{a - b:+.4f}" if a is not None and b is not None else ""
        out.append(f"  {label:<38}{_fmt(b):>9}{_fmt(a):>9}{d:>9}")
    out.append(f"  {'all OP-06 bars pass':<38}{str(mb['bars_pass']):>9}{str(ma['bars_pass']):>9}")
    out += ["", f"Examples (up to {EXAMPLES} per kind; action current -> new, gold)"]
    for k, label in KINDS.items():
        items = rep["examples"][k]
        if items:
            out.append(f"  {label} ({c[k]}): " + ", ".join(
                f"{x['id']} ({x['action'][0]} -> {x['action'][1]}, gold {x['gold_action']})" for x in items))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", type=Path, required=True, help="the changed policy file")
    ap.add_argument("--current", type=Path, default=Path(os.environ.get("TRIAGE_POLICY_PATH") or DEFAULT_POLICY),
                    help="the policy in use (default: TRIAGE_POLICY_PATH or policy/policy.toml)")
    ap.add_argument("--out", type=Path, help="also write the report as JSON")
    a = ap.parse_args(argv)
    try:
        rep = replay(a.current, a.policy)
    except PolicyError as e:
        print(f"policy file problem: {e}", file=sys.stderr)
        return 2
    print(render(rep))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
