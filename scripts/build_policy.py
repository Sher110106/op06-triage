"""Write policy/policy.toml from the permitted-action lists published in train (one list per flow).

Every train row carries its flow's published list; the script checks that all rows of a flow agree, writes
the lists sorted, adds the alias, risk tiers and routing points, then loads the file back to check it.
The routing points are a recorded decision, not a model output: experiments/routing/analyse_routing.py chose
them on train fold B by cost (experiments/routing/results_routing.json, "thresholds_from_fold_B").

  python scripts/build_policy.py [--out policy/policy.toml]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage import policy as P                                       # noqa: E402
from triage.data import load_split                                   # noqa: E402

VERSION = "2026-10-14.2"
ALIAS = {"membership-privileges": "membership"}
HIGH_RISK = ["offer-refund", "update-order", "update-account", "make-purchase", "promo-code", "make-password"]
THRESHOLDS = {"high": 0.8883, "low": 0.7104}     # Phase 2, experiments/routing/results_routing.json

HEADER = """\
# Support policy for the triage service.
#
# This file says which actions the service may suggest in each flow, which actions are risky, when to
# send a case to a person, and any extra rules. It is the only place to change the procedure: the service
# picks up a saved change within a few seconds, with no restart and no retraining.
#
# How to edit:
# - Change one thing at a time and raise `version` (for example 2026-10-14.1 to 2026-10-14.2), so the logs
#   and /healthz show which policy made each decision.
# - Use the action names exactly as they appear below. A misspelt name is an error, not a silent change.
# - Before you ship a change, run: python scripts/replay_policy.py --policy <your file>
#   It re-scores the dev conversations and counts how many suggestions, routes and warnings change.
# - If the saved file has a mistake, the service keeps the last good version and shows the error at /healthz.
#
# scripts/build_policy.py made this file from the lists published with the training data.
# Running that script again rewrites the file and drops your edits.
"""

FLOWS_NOTE = """\
# The actions each flow allows, exactly as published. Some published names (such as "n-a" or "boots") are
# never suggested by the service; they do no harm. The service favours allowed actions and marks any other
# suggestion as "off_policy".
"""

RISK_NOTE = """\
# Actions that change money, orders or accounts. Each tier has its own routing point under [thresholds].
# Actions not listed here are "low".
"""

THRESHOLD_NOTE = """\
# The service sends a case to a person when its confidence is below the point for the action's tier:
# 0.8883 for a high-risk action, 0.7104 for any other.
#
# How we chose them: we priced each outcome. A person reading a case costs 1. A wrong answer that goes out on
# its own costs 10 for a high-risk action and 3 for any other. For each tier we took the point with the lowest
# total cost on 3,966 training cases that the model did not learn from. On those cases the service sends
# 20.5% to a person, and 92.1% of the answers it gives on its own are right (97.1% for high-risk actions).
# On the 2,300 dev cases it sends 24.0%, and 91.4% of the rest are right. These figures have the LLM step on;
# with it off, the service sends 28.2% of dev cases and gets 92.0% of the rest right. Each share counts every
# case sent to a person, for any reason.
#
# Other prices give other points. On the same 3,966 cases, with reading a case at 1:
#   wrong answer costs (high-risk, other)   high     low      sent to a person   right when not sent
#   3, 2                                    0.6661   0.1295    6.2%              88.0%
#   5, 3                                    0.6661   0.7104   16.4%              91.8%
#   10, 3 (in use)                          0.8883   0.7104   20.5%              92.1%
#   10, 5                                   0.8883   0.8495   31.8%              94.9%
#   20, 10                                  0.9094   0.9071   40.2%              96.1%
# Full analysis: experiments/routing/analyse_routing.py, with its results in results_routing.json next to it.
# The LLM-off figures come from the eval run in results/raw/dev-pipeline-cpu.
"""

RULES_NOTE = """\
# Rules (none yet). A rule says an action needs one of some other actions to have happened earlier in the
# conversation. effect = "route" sends the case to a person when the rule is broken; effect = "mask" stops
# the service from suggesting the action until the rule is met. A broken rule shows up as "rule:<name>".
# Example: a refund needs the customer's identity or purchase checked first. To turn it on, remove the "# "
# from the five lines that start at [[rules]].
#
# [[rules]]
# name = "refund-needs-verification"
# action = "offer-refund"
# requires_any_observed = ["verify-identity", "validate-purchase"]
# effect = "route"
"""


def published_lists(rows: list[dict]) -> dict[str, list[str]]:
    """flow -> its permitted list; stops with a report if two rows of a flow carry different lists."""
    lists: dict[str, set[tuple[str, ...]]] = {}
    for r in rows:
        lists.setdefault(r["flow"], set()).add(tuple(r["permitted_actions"]))
    bad = {f: len(v) for f, v in lists.items() if len(v) != 1}
    if bad:
        raise SystemExit(f"permitted lists differ within a flow (flow: number of distinct lists): {bad}")
    return {f: sorted(next(iter(v))) for f, v in sorted(lists.items())}


def _toml_list(items: list[str], indent: str = "    ", width: int = 100) -> str:
    """A TOML array of strings, wrapped at `width`, one indented block between the brackets."""
    lines, cur = [], indent
    for w in (json.dumps(a) for a in items):
        if len(cur) + len(w) + 2 > width and cur.strip():
            lines.append(cur.rstrip())
            cur = indent
        cur += w + ", "
    lines.append(cur.rstrip())
    return "[\n" + "\n".join(lines) + "\n]"


def render(flows: dict[str, list[str]]) -> str:
    alias = ", ".join(f"{json.dumps(k)} = {json.dumps(v)}" for k, v in ALIAS.items())
    out = [HEADER, f'version = "{VERSION}"', "",
           '# One action, two names: the published lists say "membership-privileges", the service says "membership".',
           "# An action allowed under the first name is allowed under the second.",
           f"alias = {{ {alias} }}", "", "[flows]", FLOWS_NOTE.rstrip()]
    out += [f"{f} = {_toml_list(acts)}" for f, acts in flows.items()]
    out += ["", "[risk]", RISK_NOTE.rstrip(), f"high = {_toml_list(HIGH_RISK)}",
            "", "[thresholds]", THRESHOLD_NOTE.rstrip()]
    out += [f"{tier} = {value!r}" for tier, value in THRESHOLDS.items()]
    out += ["", RULES_NOTE.rstrip(), ""]
    return "\n".join(out)


def build(out: Path = P.DEFAULT_POLICY) -> P.Policy:
    flows = published_lists(load_split("train"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(flows), encoding="utf-8")
    return P.load(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=P.DEFAULT_POLICY)
    a = ap.parse_args()
    pol = build(a.out)
    print(f"wrote {a.out}: version {pol.version}, {len(pol.flow_permitted)} flows, "
          f"{sum(len(v) for v in pol.flow_permitted.values())} flow-action pairs, {len(pol.rules)} rules")
