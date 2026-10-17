"""The support policy, read from policy/policy.toml: permitted actions per flow, risk tiers, routing points, rules.

`load(path)` reads and checks the file and raises PolicyError with a plain message on any mistake.
`PolicyStore` holds the current policy for the service and reloads it when the file changes; a bad file
keeps the last good policy and records the error for /healthz.
"""
from __future__ import annotations

import difflib
import math
import os
import re
import threading
import time
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data import ROOT
from .text import ACTIONS30

DEFAULT_POLICY = ROOT / "policy" / "policy.toml"
ACTIONS = frozenset(ACTIONS30)
FLOWS = frozenset({"account_access", "manage_account", "order_issue", "product_defect", "purchase_dispute",
                   "shipping_issue", "single_item_query", "storewide_query", "subscription_inquiry",
                   "troubleshoot_site"})
# names in the published permitted lists that are never a next-action label; allowed in [flows] only
LIST_ONLY = frozenset({"boots", "end-conversation", "enter-detail", "jacket", "jeans", "n-a", "notify-internal-team",
                       "policy", "pricing", "select-answer", "shirt", "timing"})
EFFECTS = ("route", "mask")
_TOP_KEYS = {"version", "alias", "flows", "risk", "thresholds", "rules", "permission_review"}
_RULE_KEYS = {"name", "action", "requires_any_observed", "effect"}
_RULE_NAME = re.compile(r"[A-Za-z0-9_-]+")             # the name shows up in flags and metrics as "rule:<name>"


class PolicyError(ValueError):
    """The policy file is missing, not valid TOML, or breaks a rule; the message says what to fix."""


@dataclass(frozen=True)
class Rule:
    name: str
    action: str                              # alias applied
    requires_any_observed: tuple[str, ...]   # alias applied
    effect: str                              # "route" or "mask"


@dataclass(frozen=True)
class Policy:
    version: str
    alias: dict[str, str]
    flow_permitted: dict[str, frozenset[str]]   # published names plus their alias targets
    risk: dict[str, str]                        # action (alias applied) -> tier; unlisted actions are "low"
    thresholds: dict[str, float]
    rules: tuple[Rule, ...]
    permission_review: dict[str, float] = field(default_factory=dict)

    def _canon(self, action: str) -> str:
        return self.alias.get(action, action)

    def _permitted(self, action: str, flow: str) -> bool:
        p = self.flow_permitted.get(flow, frozenset())
        return action in p or self._canon(action) in p

    def _unmet(self, rule: Rule, seen: set[str]) -> bool:
        return bool(rule.requires_any_observed) and not seen.intersection(rule.requires_any_observed)

    def prior_matrix(self, flows: Sequence[str], actions: Sequence[str]) -> np.ndarray:
        """float32 table, flows x actions: 1 where the flow permits the action, else 0."""
        unknown = [f for f in flows if f not in self.flow_permitted]
        if unknown:
            raise PolicyError(f"the policy has no flow named {unknown[0]!r}")
        return np.array([[self._permitted(a, f) for a in actions] for f in flows], dtype=np.float32)

    def masked_actions(self, observed: Sequence[str]) -> frozenset[str]:
        """Actions that "mask" rules remove because nothing they require has been seen yet."""
        seen = {self._canon(o) for o in observed}
        return frozenset(r.action for r in self.rules if r.effect == "mask" and self._unmet(r, seen))

    def violations(self, action: str, flow: str, observed: Sequence[str]) -> list[str]:
        """Flags for a chosen action: "off_policy" if the flow does not permit it, "rule:<name>" per broken rule."""
        out = [] if self._permitted(action, flow) else ["off_policy"]
        seen, act = {self._canon(o) for o in observed}, self._canon(action)
        out += [f"rule:{r.name}" for r in self.rules if r.action == act and self._unmet(r, seen)]
        return out

    def threshold_for(self, action: str) -> float:
        return self.thresholds[self.risk.get(self._canon(action), "low")]

    def minimum_permission_support(self, action: str) -> float:
        """Minimum inferred-flow permission mass; zero disables this additional review rule."""
        return self.permission_review.get(self.risk.get(self._canon(action), "low"), 0.0)


# ----------------------------------------------------------------------------- loading and checks

def _names(value, where: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise PolicyError(f"{where} must be a list of action names in quotes")
    return value


def _check_name(name: str, where: str, allowed: frozenset[str]) -> None:
    if name not in allowed:
        near = difflib.get_close_matches(name, sorted(allowed), n=1)
        hint = f"; did you mean {near[0]!r}?" if near else ""
        raise PolicyError(f"{where}: unknown action {name!r}{hint}")


def _parse(data: dict) -> Policy:
    extra = sorted(set(data) - _TOP_KEYS)
    if extra:
        raise PolicyError(f"unknown setting {extra[0]!r}; allowed: {', '.join(sorted(_TOP_KEYS))}")

    version = data.get("version")
    if not isinstance(version, str) or not version.strip():
        raise PolicyError('version must be set, as text in quotes, for example version = "2026-10-14.1"')

    alias = data.get("alias", {})
    if not isinstance(alias, dict):
        raise PolicyError('alias must be a table, for example alias = { "membership-privileges" = "membership" }')
    for k, v in alias.items():
        if not isinstance(v, str):
            raise PolicyError(f"alias {k!r} must point to an action name in quotes")
        _check_name(v, f"alias {k!r}", ACTIONS)
    known = ACTIONS | frozenset(alias)

    flows = data.get("flows")
    if not isinstance(flows, dict):
        raise PolicyError("the [flows] section is missing")
    for f in sorted(flows):
        if f not in FLOWS:
            raise PolicyError(f"unknown flow {f!r}; the flows are: {', '.join(sorted(FLOWS))}")
    missing = sorted(FLOWS - set(flows))
    if missing:
        raise PolicyError(f"flow {missing[0]!r} is missing from [flows]; every flow must be listed")
    flow_permitted = {}
    for f, acts in flows.items():
        for a in _names(acts, f"flow {f!r}"):
            _check_name(a, f"flow {f!r}", known | LIST_ONLY)
        flow_permitted[f] = frozenset(acts) | {alias[a] for a in acts if a in alias}

    thresholds = data.get("thresholds")
    if not isinstance(thresholds, dict):
        raise PolicyError("the [thresholds] section is missing")
    for tier, v in thresholds.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or not 0 <= v <= 1:
            raise PolicyError(f"threshold {tier!r} must be a number from 0 to 1, got {v!r}")
    if "low" not in thresholds:
        raise PolicyError('[thresholds] needs a "low" value; it applies to every action not listed in [risk]')

    risk_tiers = data.get("risk", {})
    if not isinstance(risk_tiers, dict):
        raise PolicyError("[risk] must be a section of tier = [actions]")
    risk: dict[str, str] = {}
    for tier, acts in risk_tiers.items():
        if tier not in thresholds:
            raise PolicyError(f"risk tier {tier!r} has no threshold; add {tier} = <number> under [thresholds]")
        for a in _names(acts, f"risk tier {tier!r}"):
            _check_name(a, f"risk tier {tier!r}", known)
            a = alias.get(a, a)
            if risk.get(a, tier) != tier:
                raise PolicyError(f"action {a!r} is in two risk tiers: {risk[a]!r} and {tier!r}")
            risk[a] = tier

    permission_review = data.get("permission_review", {})
    if not isinstance(permission_review, dict):
        raise PolicyError("permission_review must be a table of risk tier = minimum permitted probability")
    for tier, value in permission_review.items():
        if tier not in thresholds:
            raise PolicyError(f"permission_review tier {tier!r} has no threshold")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise PolicyError(f"permission_review {tier!r} must be a number from 0 to 1")

    raw_rules = data.get("rules", [])
    if not isinstance(raw_rules, list) or not all(isinstance(r, dict) for r in raw_rules):
        raise PolicyError("rules must be written as [[rules]] sections")
    rules, names = [], set()
    for i, r in enumerate(raw_rules, 1):
        name = r.get("name")
        if not isinstance(name, str) or not _RULE_NAME.fullmatch(name):
            raise PolicyError(f"rule {i} needs a name in quotes, made of letters, digits, - and _ only")
        where = f"rule {name!r}"
        extra = sorted(set(r) - _RULE_KEYS)
        if extra:
            raise PolicyError(f"{where}: unknown setting {extra[0]!r}; allowed: {', '.join(sorted(_RULE_KEYS))}")
        if name in names:
            raise PolicyError(f"two rules are named {name!r}; rule names must differ")
        names.add(name)
        action = r.get("action")
        if not isinstance(action, str):
            raise PolicyError(f"{where} needs an action in quotes")
        _check_name(action, where, known)
        req = _names(r.get("requires_any_observed", []), f"{where} requires_any_observed")
        for a in req:
            _check_name(a, f"{where} requires_any_observed", known)
        if r.get("effect") not in EFFECTS:
            raise PolicyError(f'{where}: effect must be "route" or "mask", got {r.get("effect")!r}')
        rules.append(Rule(name, alias.get(action, action), tuple(alias.get(a, a) for a in req), r["effect"]))

    return Policy(version=version, alias=dict(alias), flow_permitted=flow_permitted, risk=risk,
                  thresholds={k: float(v) for k, v in thresholds.items()}, rules=tuple(rules),
                  permission_review={k: float(v) for k, v in permission_review.items()})


def loads(text: str) -> Policy:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise PolicyError(f"the file is not valid TOML: {e}") from None
    return _parse(data)


def load(path=DEFAULT_POLICY) -> Policy:
    try:
        text = Path(path).read_bytes().decode("utf-8")
    except OSError as e:
        raise PolicyError(f"cannot read the policy file {str(path)!r}: {e.strerror or e}") from None
    except UnicodeDecodeError:
        raise PolicyError(f"the policy file {str(path)!r} is not UTF-8 text") from None
    return loads(text)


class PolicyStore:
    """The current policy, shared across threads. `current()` reloads when the file's mtime or size changes,
    checked at most every `check_every_s` seconds. A bad file keeps the last good policy and sets `last_error`."""

    def __init__(self, path=DEFAULT_POLICY, check_every_s: float = 2.0):
        self.path = Path(path)
        self.check_every_s = check_every_s
        self._lock = threading.Lock()
        self._sig = self._signature()                  # taken before the read, so a write during it is seen later
        self._policy = load(self.path)                 # a bad file at start is an error: there is no last good one
        self.loaded_at = time.time()
        self.last_error: str | None = None
        self._checked = time.monotonic()

    @property
    def version(self) -> str:
        return self._policy.version

    def _signature(self) -> tuple[int, int] | None:
        try:
            st = os.stat(self.path)
        except OSError:
            return None
        return st.st_mtime_ns, st.st_size

    def current(self) -> Policy:
        if time.monotonic() - self._checked < self.check_every_s:
            return self._policy
        with self._lock:
            if time.monotonic() - self._checked >= self.check_every_s:
                self._reload_if_changed()
                self._checked = time.monotonic()
            return self._policy

    def _reload_if_changed(self) -> None:
        sig = self._signature()
        if sig == self._sig:
            return
        self._sig = sig
        try:
            policy = load(self.path)
        except PolicyError as e:
            self.last_error = str(e)
            return
        self._policy, self.loaded_at, self.last_error = policy, time.time(), None
