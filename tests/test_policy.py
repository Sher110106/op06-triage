"""Tests for triage/policy.py, policy/policy.toml and scripts/build_policy.py."""
from __future__ import annotations

import importlib.util
import os
import re
import threading
import tomllib

import numpy as np
import pytest

from triage import policy as P
from triage.data import ROOT, load_split
from triage.policy import Policy, PolicyError, PolicyStore, load, loads

BUNDLE = ROOT / "models" / "classical_v0.joblib"
POLICY = ROOT / "policy" / "policy.toml"

FLOW_LINES = {f: '["pull-up-account", "verify-identity"]' for f in sorted(P.FLOWS)}
FLOW_LINES["order_issue"] = '["membership-privileges", "offer-refund", "pull-up-account", "verify-identity"]'
REFUND_RULE = """
[[rules]]
name = "refund-needs-verification"
action = "offer-refund"
requires_any_observed = ["verify-identity", "validate-purchase"]
effect = "{effect}"
"""


def policy_text(version: str = "t.1", rules: str = "") -> str:
    flows = "\n".join(f"{f} = {acts}" for f, acts in FLOW_LINES.items())
    return (f'version = "{version}"\nalias = {{ "membership-privileges" = "membership" }}\n'
            f'[flows]\n{flows}\n[risk]\nhigh = ["offer-refund", "update-order"]\n'
            f"[thresholds]\nhigh = 0.9\nlow = 0.5\n{rules}")


def rewrite(path, text: str) -> None:
    """Write and move the mtime forward a full second, so the change is seen whatever the clock resolution."""
    before = path.stat().st_mtime_ns
    path.write_text(text, encoding="utf-8")
    os.utime(path, ns=(before + 10**9, before + 10**9))


@pytest.fixture(scope="session")
def bundle():
    if not BUNDLE.exists():
        pytest.skip("models/classical_v0.joblib is not built; run scripts/reproduce.sh train")
    import joblib
    return joblib.load(BUNDLE)


@pytest.fixture(scope="session")
def train_rows():
    return load_split("train")


@pytest.fixture(scope="session")
def shipped() -> Policy:
    return load(POLICY)


@pytest.fixture()
def tmp_policy(tmp_path):
    p = tmp_path / "policy.toml"
    p.write_text(policy_text(), encoding="utf-8")
    return p


# ----------------------------------------------------------------------------- the shipped file

def test_shipped_file_loads(shipped):
    assert shipped.version == "2026-10-14.1"
    assert set(shipped.flow_permitted) == P.FLOWS
    assert shipped.alias == {"membership-privileges": "membership"}
    assert set(shipped.risk) == {"offer-refund", "update-order", "update-account", "make-purchase", "promo-code",
                                 "make-password"}
    assert set(shipped.risk.values()) == {"high"}
    assert shipped.thresholds == {"high": 0.8498168103175511, "low": 0.8498168103175511}
    assert shipped.rules == ()


def test_shipped_thresholds_equal_the_bundle_routing_point(shipped, bundle):
    """Written in full: a rounded 0.8498 routed confidences in [0.8498, 0.8498168) differently from v0."""
    assert set(shipped.thresholds.values()) == {bundle["route_threshold"]}
    assert _build_module().ROUTE_THRESHOLD == bundle["route_threshold"]


def test_prior_matrix_equals_bundle_table(shipped, bundle):
    m = shipped.prior_matrix(bundle["fsvc"].classes_, bundle["acts"])
    assert m.dtype == np.float32 and m.shape == bundle["perm"].shape
    assert np.array_equal(m, bundle["perm"])
    assert shipped.flow_permitted == {f: frozenset(v) for f, v in bundle["flow_perm"].items()}


def test_name_lists_match_the_data(train_rows, bundle):
    assert P.ACTIONS == {r["action"] for r in train_rows} == set(bundle["acts"])
    assert P.FLOWS == {r["flow"] for r in train_rows} == set(bundle["fsvc"].classes_)
    published = {a for r in train_rows for a in r["permitted_actions"]}
    assert published - P.ACTIONS == P.LIST_ONLY | {"membership-privileges"}


def test_alias(shipped):
    published = tomllib.loads(POLICY.read_text())["flows"]["order_issue"]
    assert "membership-privileges" in published and "membership" not in published
    assert shipped.violations("membership", "order_issue", []) == []
    assert shipped.violations("membership-privileges", "order_issue", []) == []
    assert shipped.prior_matrix(["order_issue"], ["membership", "membership-privileges"]).tolist() == [[1.0, 1.0]]
    assert shipped.violations("membership", "account_access", []) == ["off_policy"]


def test_off_policy(shipped):
    assert shipped.violations("offer-refund", "order_issue", []) == []
    assert shipped.violations("make-password", "order_issue", []) == ["off_policy"]
    assert shipped.violations("offer-refund", "no_such_flow", []) == ["off_policy"]
    with pytest.raises(PolicyError, match="no flow named 'no_such_flow'"):
        shipped.prior_matrix(["no_such_flow"], ["offer-refund"])


def test_threshold_for(tmp_policy):
    pol = load(tmp_policy)
    assert pol.threshold_for("offer-refund") == 0.9
    assert pol.threshold_for("search-faq") == 0.5


def test_example_rule_in_shipped_file_is_valid():
    lines = POLICY.read_text().splitlines()
    start = lines.index("# [[rules]]")
    text = "\n".join(lines[:start] + [ln.removeprefix("# ") for ln in lines[start:start + 5]])
    pol = loads(text)
    assert [r.name for r in pol.rules] == ["refund-needs-verification"]
    assert pol.violations("offer-refund", "order_issue", ["pull-up-account"]) == ["rule:refund-needs-verification"]


# ----------------------------------------------------------------------------- rules

def test_route_rule(tmp_path):
    p = tmp_path / "route.toml"
    p.write_text(policy_text(rules=REFUND_RULE.format(effect="route")))
    pol = load(p)
    assert pol.violations("offer-refund", "order_issue", ["pull-up-account"]) == ["rule:refund-needs-verification"]
    assert pol.violations("offer-refund", "order_issue", ["pull-up-account", "verify-identity"]) == []
    assert pol.violations("pull-up-account", "order_issue", []) == []
    assert pol.violations("offer-refund", "account_access", []) == ["off_policy", "rule:refund-needs-verification"]
    assert pol.masked_actions([]) == frozenset()


def test_mask_rule(tmp_path):
    p = tmp_path / "mask.toml"
    p.write_text(policy_text(rules=REFUND_RULE.format(effect="mask")))
    pol = load(p)
    assert pol.masked_actions(["pull-up-account"]) == {"offer-refund"}
    assert pol.masked_actions(["validate-purchase"]) == frozenset()
    assert pol.violations("offer-refund", "order_issue", []) == ["rule:refund-needs-verification"]


def test_rule_names_go_through_the_alias():
    rule = ('[[rules]]\nname = "m"\naction = "membership-privileges"\n'
            'requires_any_observed = ["pull-up-account"]\neffect = "mask"\n')
    pol = loads(policy_text(rules=rule))
    assert pol.masked_actions([]) == {"membership"}
    assert pol.violations("membership", "order_issue", []) == ["rule:m"]
    assert pol.violations("membership", "order_issue", ["pull-up-account"]) == []


def test_rule_without_requirement_never_fires():
    pol = loads(policy_text(rules='[[rules]]\nname = "n"\naction = "offer-refund"\neffect = "mask"\n'))
    assert pol.masked_actions([]) == frozenset()
    assert pol.violations("offer-refund", "order_issue", []) == []


# ----------------------------------------------------------------------------- validation

def _rule(**kw) -> str:
    r = {"name": '"r"', "action": '"offer-refund"', "requires_any_observed": '["verify-identity"]',
         "effect": '"route"'} | kw
    return "[[rules]]\n" + "".join(f"{k} = {v}\n" for k, v in r.items() if v is not None)


BAD = [
    ("misspelt action in a flow", lambda t: t.replace('"offer-refund", "pull', '"ofer-refund", "pull'),
     "flow 'order_issue': unknown action 'ofer-refund'; did you mean 'offer-refund'"),
    ("unknown flow", lambda t: t.replace("troubleshoot_site =", "troubleshooting ="), "unknown flow 'troubleshooting'"),
    ("missing flow", lambda t: t.replace("troubleshoot_site =", "#"), "flow 'troubleshoot_site' is missing"),
    ("no flows", lambda t: t.replace("[flows]", "[flowz]"), "unknown setting 'flowz'"),
    ("threshold above 1", lambda t: t.replace("low = 0.5", "low = 1.5"), "threshold 'low' must be a number from 0 to 1"),
    ("threshold below 0", lambda t: t.replace("high = 0.9", "high = -0.1"), "threshold 'high' must be a number"),
    ("threshold as text", lambda t: t.replace("low = 0.5", 'low = "0.5"'), "threshold 'low' must be a number"),
    ("threshold nan", lambda t: t.replace("low = 0.5", "low = nan"), "threshold 'low' must be a number"),
    ("tier without threshold", lambda t: t.replace("[thresholds]", 'medium = ["promo-code"]\n[thresholds]'),
     "risk tier 'medium' has no threshold"),
    ("no low threshold", lambda t: t.replace("low = 0.5", ""), 'needs a "low" value'),
    ("action in two tiers", lambda t: t.replace("[thresholds]", 'low = ["offer-refund"]\n[thresholds]'),
     "'offer-refund' is in two risk tiers"),
    ("unknown risk action", lambda t: t.replace('high = ["offer-refund"', 'high = ["refund"'),
     "risk tier 'high': unknown action 'refund'"),
    ("list-only name in risk", lambda t: t.replace('high = ["offer-refund"', 'high = ["n-a"'), "unknown action 'n-a'"),
    ("empty version", lambda t: t.replace('version = "t.1"', 'version = ""'), "version must be set"),
    ("no version", lambda t: t.replace('version = "t.1"', ""), "version must be set"),
    ("version as a date", lambda t: t.replace('version = "t.1"', "version = 2026-10-14"), "version must be set"),
    ("bad alias target", lambda t: t.replace('= "membership" }', '= "membershp" }'),
     "alias 'membership-privileges': unknown action 'membershp'"),
    ("not TOML", lambda t: t + "\nhigh = [", "not valid TOML"),
    ("unknown setting", lambda t: "treshold = 1\n" + t, "unknown setting 'treshold'"),
    ("flow list not a list", lambda t: t.replace("troubleshoot_site = [", 'troubleshoot_site = "x" #'),
     "flow 'troubleshoot_site' must be a list"),
    ("bad effect", lambda t: t + _rule(effect='"block"'), "rule 'r': effect must be \"route\" or \"mask\""),
    ("no effect", lambda t: t + _rule(effect=None), "rule 'r': effect must be"),
    ("unknown rule action", lambda t: t + _rule(action='"refund"'), "rule 'r': unknown action 'refund'"),
    ("list-only rule action", lambda t: t + _rule(action='"n-a"'), "rule 'r': unknown action 'n-a'"),
    ("unknown required action", lambda t: t + _rule(requires_any_observed='["verify-idenity"]'),
     "rule 'r' requires_any_observed: unknown action 'verify-idenity'; did you mean 'verify-identity'"),
    ("rule without name", lambda t: t + _rule(name=None), "rule 1 needs a name"),
    ("rule name with spaces", lambda t: t + _rule(name='"refund check"'), "rule 1 needs a name"),
    ("misspelt rule setting", lambda t: t + _rule(requires_any_observed=None, requires_any_observd='["verify-identity"]'),
     "rule 'r': unknown setting 'requires_any_observd'"),
    ("two rules with one name", lambda t: t + _rule() + _rule(), "two rules are named 'r'"),
]


@pytest.mark.parametrize("edit,message", [(e, m) for _, e, m in BAD], ids=[n for n, _, _ in BAD])
def test_validation_errors(edit, message):
    text = edit(policy_text())
    assert text != policy_text()
    with pytest.raises(PolicyError, match=re.escape(message)):
        loads(text)


def test_missing_file(tmp_path):
    with pytest.raises(PolicyError, match="cannot read the policy file"):
        load(tmp_path / "nope.toml")


# ----------------------------------------------------------------------------- PolicyStore

def test_hot_reload(tmp_policy):
    store = PolicyStore(tmp_policy, check_every_s=0.0)
    assert store.current().version == "t.1" and store.version == "t.1"
    first = store.loaded_at
    rewrite(tmp_policy, policy_text("t.2", rules=REFUND_RULE.format(effect="mask")))
    pol = store.current()
    assert pol.version == "t.2" and store.version == "t.2"
    assert pol.masked_actions([]) == {"offer-refund"}
    assert store.loaded_at >= first and store.last_error is None
    assert store.path == tmp_policy


def test_checks_at_most_every_interval(tmp_policy):
    store = PolicyStore(tmp_policy, check_every_s=3600)
    rewrite(tmp_policy, policy_text("t.2"))
    assert store.current().version == "t.1"


def test_bad_file_keeps_last_good(tmp_policy):
    store = PolicyStore(tmp_policy, check_every_s=0.0)
    good = store.current()
    rewrite(tmp_policy, "version = ")
    assert store.current() is good and store.version == "t.1"
    assert "not valid TOML" in store.last_error
    rewrite(tmp_policy, policy_text("t.2").replace('"offer-refund", "pull', '"ofer-refund", "pull'))
    assert store.current() is good
    assert "unknown action 'ofer-refund'" in store.last_error
    rewrite(tmp_policy, policy_text("t.3"))
    assert store.current().version == "t.3" and store.last_error is None


def test_deleted_file_keeps_last_good(tmp_policy):
    store = PolicyStore(tmp_policy, check_every_s=0.0)
    text = tmp_policy.read_text()
    tmp_policy.unlink()
    assert store.current().version == "t.1"
    assert "cannot read the policy file" in store.last_error
    tmp_policy.write_text(text.replace('"t.1"', '"t.2"'))
    assert store.current().version == "t.2" and store.last_error is None


def test_bad_file_at_start_raises(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("version = 1")
    with pytest.raises(PolicyError):
        PolicyStore(p)


def test_store_is_thread_safe(tmp_policy):
    store = PolicyStore(tmp_policy, check_every_s=0.0)
    seen, errors = set(), []

    def reader():
        try:
            for _ in range(300):
                seen.add(store.current().version)
        except Exception as e:                                         # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(8)]
    for t in threads:
        t.start()
    for i in range(20):
        rewrite(tmp_policy, policy_text(f"t.{i % 2 + 1}"))
    for t in threads:
        t.join()
    assert not errors and seen <= {"t.1", "t.2"}
    assert store.current().version == "t.2" and store.last_error is None


# ----------------------------------------------------------------------------- build script

def _build_module():
    spec = importlib.util.spec_from_file_location("build_policy", ROOT / "scripts" / "build_policy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_script(tmp_path, train_rows, bundle):
    mod = _build_module()
    pol = mod.build(tmp_path / "policy.toml")
    assert (tmp_path / "policy.toml").read_bytes() == P.DEFAULT_POLICY.read_bytes()      # the shipped file, unedited
    assert set(pol.thresholds.values()) == {bundle["route_threshold"]}
    assert set(mod.build(tmp_path / "nb.toml", tmp_path / "none.joblib").thresholds.values()) == {mod.ROUTE_THRESHOLD}
    assert np.array_equal(pol.prior_matrix(bundle["fsvc"].classes_, bundle["acts"]), bundle["perm"])
    published = {r["flow"]: frozenset(r["permitted_actions"]) for r in train_rows}
    expected = {f: v | ({"membership"} if "membership-privileges" in v else set()) for f, v in published.items()}
    assert pol.flow_permitted == expected


def test_build_script_rejects_disagreeing_lists():
    rows = [{"flow": "order_issue", "permitted_actions": ["offer-refund"]},
            {"flow": "order_issue", "permitted_actions": ["update-order"]}]
    with pytest.raises(SystemExit, match="differ within a flow"):
        _build_module().published_lists(rows)
