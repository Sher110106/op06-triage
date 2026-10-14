"""Write policy/policy.toml from the permitted-action lists published in train (one list per flow).

Every train row carries its flow's published list; the script checks that all rows of a flow agree, writes
the lists sorted, adds the alias, risk tiers and placeholder thresholds, then loads the file back to check it.
Both thresholds are the CPU model's own routing point (`route_threshold` in the bundle, written in full), so with
the LLM step off the service routes exactly as the v0 model does.

  python scripts/build_policy.py [--out policy/policy.toml] [--bundle models/classical_v0.joblib]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage import policy as P                                       # noqa: E402
from triage.classical import DEFAULT_BUNDLE                          # noqa: E402
from triage.data import load_split                                   # noqa: E402

VERSION = "2026-10-14.1"
ALIAS = {"membership-privileges": "membership"}
HIGH_RISK = ["offer-refund", "update-order", "update-account", "make-purchase", "promo-code", "make-password"]
ROUTE_THRESHOLD = 0.8498168103175511     # route_threshold of the reference bundle; used when no bundle is built

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
# The service sends a case to a person when its confidence is below the point for the action's tier.
# Phase 2 will set real values. For now both tiers use the v0 model's routing point (written in full, so the
# service routes exactly as that model does), and high-risk actions get no extra care yet.
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


def route_threshold(bundle: Path = DEFAULT_BUNDLE) -> float:
    """The CPU model's routing point: from the bundle when it is built, else the reference bundle's value."""
    if not Path(bundle).exists():
        return ROUTE_THRESHOLD
    import joblib
    return float(joblib.load(bundle)["route_threshold"])


def render(flows: dict[str, list[str]], threshold: float = ROUTE_THRESHOLD) -> str:
    alias = ", ".join(f"{json.dumps(k)} = {json.dumps(v)}" for k, v in ALIAS.items())
    out = [HEADER, f'version = "{VERSION}"', "",
           '# One action, two names: the published lists say "membership-privileges", the service says "membership".',
           "# An action allowed under the first name is allowed under the second.",
           f"alias = {{ {alias} }}", "", "[flows]", FLOWS_NOTE.rstrip()]
    out += [f"{f} = {_toml_list(acts)}" for f, acts in flows.items()]
    out += ["", "[risk]", RISK_NOTE.rstrip(), f"high = {_toml_list(HIGH_RISK)}",
            "", "[thresholds]", THRESHOLD_NOTE.rstrip()]
    out += [f"{tier} = {threshold!r}" for tier in ("high", "low")]
    out += ["", RULES_NOTE.rstrip(), ""]
    return "\n".join(out)


def build(out: Path = P.DEFAULT_POLICY, bundle: Path = DEFAULT_BUNDLE) -> P.Policy:
    flows = published_lists(load_split("train"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(flows, route_threshold(bundle)), encoding="utf-8")
    return P.load(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=P.DEFAULT_POLICY)
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    a = ap.parse_args()
    pol = build(a.out, a.bundle)
    print(f"wrote {a.out}: version {pol.version}, {len(pol.flow_permitted)} flows, "
          f"{sum(len(v) for v in pol.flow_permitted.values())} flow-action pairs, {len(pol.rules)} rules")
