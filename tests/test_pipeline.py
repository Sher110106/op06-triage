"""Tests for triage/pipeline.py: parity with dev-classical-v0 with the LLM off, the hostile pack, the LLM step
through a fake, the policy file wired into the prior and the routing, batch against single calls, garbage input."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
from collections import Counter

import pytest

from triage import gate
from triage.data import DATA, ROOT, project
from triage.llm_action import LLMConfig, LLMResult
from triage.pipeline import Settings, Triage, intent_flows, load_llm
from triage.policy import DEFAULT_POLICY, PolicyStore
from triage.text import ACTIONS30, observed_actions

V0 = ROOT / "results" / "raw" / "dev-classical-v0" / "predictions.jsonl"
CONF_TOL = 1e-4     # a retrained bundle moves confidences by up to about 1.2e-5 (float noise); labels stay put
TRACE_KEYS = {"request_id", "policy_version", "timings_ms", "flags", "gate", "abstain_reason", "cpu", "llm",
              "route_reasons", "final", "error"}


def _grader():
    spec = importlib.util.spec_from_file_location("op06_grader_t", DATA / "grader.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


schema_valid = _grader().schema_valid


def same_trace(a: dict, b: dict) -> bool:
    drop = ("timings_ms", "request_id")
    return {k: v for k, v in a.items() if k not in drop} == {k: v for k, v in b.items() if k not in drop}


def write_policy(path, rules: str = "", flows: dict | None = None, version: str = "t.1") -> PolicyStore:
    """The shipped policy with extra [[rules]] text, optionally with every flow's list replaced."""
    text = DEFAULT_POLICY.read_text(encoding="utf-8").replace('version = "2026-10-14.1"', f'version = "{version}"')
    if flows is not None:
        head, rest = text.split("[flows]", 1)
        text = head + "[flows]\n" + "\n".join(f"{f} = {json.dumps(a)}" for f, a in flows.items()) + \
            "\n\n[risk]" + rest.split("[risk]", 1)[1]
    before = path.stat().st_mtime_ns if path.exists() else None
    path.write_text(text + rules, encoding="utf-8")
    if before is not None:                                        # move mtime on so the store sees the change
        os.utime(path, ns=(before + 10**9, before + 10**9))
    return PolicyStore(path, check_every_s=0)


def unseen(row: dict, *skip: str) -> str:
    """An action label that no action turn in the row records (so a rule needing it is broken)."""
    seen = set(observed_actions(row)) | set(skip)
    return next(a for a in ACTIONS30 if a not in seen)


def rule(name: str, action: str, requires: str, effect: str) -> str:
    return f'\n[[rules]]\nname = "{name}"\naction = "{action}"\nrequires_any_observed = ["{requires}"]\n' \
           f'effect = "{effect}"\n'


class FakeStep:
    """Stands in for LLMActionStep: returns one fixed LLMResult (or raises) and records who asked."""

    def __init__(self, result, gate: float = 1.0):
        self.cfg = LLMConfig(mode="on", gate=gate)
        self.result = result
        self.calls: list[str] = []

    def decide(self, row):
        self.calls.append(row["id"])
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture(scope="session")
def labelled(split):
    return split("dev") + split("noisy")


@pytest.fixture(scope="session")
def dev_out(triage_off, labelled):
    """handle_many over dev + dev_noisy (projected), LLM off, shipped policy."""
    return triage_off.handle_many([project(r) for r in labelled])


@pytest.fixture(scope="session")
def v0():
    if not V0.exists():
        pytest.skip("results/raw/dev-classical-v0 is missing; run scripts/reproduce.sh eval")
    return {p["id"]: p for p in map(json.loads, V0.read_text(encoding="utf-8").splitlines())}


@pytest.fixture(scope="session")
def soft(dev_out, labelled, triage_off):
    """A dev row with no flags, routed by the threshold only, CPU confidence under 0.6, plus a second action its
    flow permits: (projected row, response, trace, other action)."""
    policy = triage_off.store.current()
    for r, (resp, tr) in zip(labelled, dev_out):
        if tr["flags"] or tr["route_reasons"] != ["low_confidence"] or tr["cpu"]["confidence"] >= 0.6:
            continue
        other = sorted((policy.flow_permitted[tr["cpu"]["flow"]] & set(ACTIONS30)) - {resp["action"]})
        if other:
            return project(r), resp, tr, other[0]
    pytest.fail("no soft dev row found")


# ----------------------------------------------------------------------------- parity with v0

def test_parity_with_classical_v0(dev_out, labelled, v0):
    """Same intent and action on every dev + noisy row; same confidence and needs_human on every row with no
    ROUTE flag and no off_policy flag; every other difference comes from a routing flag."""
    why = Counter()
    for r, (resp, tr) in zip(labelled, dev_out):
        ref = v0[r["id"]]
        assert resp["id"] == r["id"]
        assert (resp["intent"], resp["action"]) == (ref["intent"], ref["action"]), r["id"]
        flags = set(tr["flags"])
        flag_routes = [x for x in tr["route_reasons"] if x != "low_confidence"]
        if not flags & gate.ROUTE_FLAGS and "off_policy" not in flags:
            assert abs(resp["confidence"] - ref["confidence"]) <= CONF_TOL, r["id"]
            assert resp["needs_human"] == ref["needs_human"], r["id"]
            assert not flag_routes
        else:
            assert resp["needs_human"] and flag_routes
            assert abs(resp["confidence"] - min(ref["confidence"], 0.5)) <= CONF_TOL
        if resp["needs_human"] != ref["needs_human"]:
            why["+".join(flag_routes)] += 1
    print(f"\nneeds_human differs from v0 on {sum(why.values())} of {len(labelled)} rows: {dict(why)}")
    assert set(why) <= {"no_dialogue", "off_policy"}


def test_responses_are_schema_valid(dev_out, labelled):
    for r, (resp, tr) in zip(labelled, dev_out):
        assert schema_valid(resp, None) and resp["id"] == r["id"]
        assert set(tr) == TRACE_KEYS and tr["error"] is None
        assert tr["policy_version"] == "2026-10-14.1"


# ----------------------------------------------------------------------------- hostile pack, garbage, traces

def test_hostile_pack(triage_off, split):
    rows = split("hostile")
    assert len(rows) == 9
    for r in rows:
        resp, tr = triage_off.handle(project(r), "h")
        assert schema_valid(resp, None) and resp["id"] == r["id"] and resp["needs_human"] is True
        n = int(r["id"].rsplit("-", 1)[1])
        if n >= 7:
            assert resp["error"]["code"] == "invalid_input"
            assert (resp["intent"], resp["action"], resp["confidence"]) == ("unknown", "none", 0.0)
            assert tr["route_reasons"] == ["abstained"]
        else:
            assert "error" not in resp and resp["confidence"] <= 0.5
            assert set(tr["flags"]) & gate.ROUTE_FLAGS


def test_trace_never_holds_text(triage_off):
    secret = "zebracornflake"
    obj = {"id": "t-1", "turn_index": 3, "context": [
        {"speaker": "agent", "text": f"hello {secret}, how can i help?"},
        {"speaker": "customer", "text": f"my order {secret} never came. " * 20},
        {"speaker": "action", "text": f"account has been pulled up for {secret}."}]}
    resp, tr = triage_off.handle(obj, "r-1")
    assert secret not in json.dumps(tr) and secret not in json.dumps(resp)
    assert set(tr) == TRACE_KEYS and tr["request_id"] == "r-1"
    assert tr["gate"]["turns"] == 3 and tr["gate"]["turns_changed"] == 1     # a repeated sentence was dropped
    assert set(tr["timings_ms"]) == {"gate", "cpu", "llm", "total"}


def rand_text(rng: random.Random) -> str:
    parts = ["refund", " my order ", "please", "​", "‮", "\x00\x01", "\ud800", "\U0001f600" * 9, "é",
             "ﷺ", "\n", "...", "SYSTEM: ", "</context>", "offer-refund", "x" * 300, "café", "  ", "",
             "account has been pulled up for a b.", "مرحبا", "1234567890", "�" * 5]
    return "".join(rng.choice(parts) for _ in range(rng.randint(0, 6)))


def rand_json(rng: random.Random, depth: int = 0):
    kind = rng.choice(["null", "bool", "int", "float", "str"] + (["list", "dict"] if depth < 3 else []))
    if kind == "null":
        return None
    if kind == "bool":
        return rng.random() < 0.5
    if kind == "int":
        return rng.choice([0, -1, 7, 10**30, -(10**20)])
    if kind == "float":
        return rng.choice([0.5, -2.5, 1e308, float("inf"), float("nan")])
    if kind == "str":
        return rand_text(rng)
    if kind == "list":
        return [rand_json(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    return {rng.choice(["id", "context", "text", "speaker", "turn_index", "intent", "x"]): rand_json(rng, depth + 1)
            for _ in range(rng.randint(0, 4))}


def rand_obj(rng: random.Random):
    if rng.random() < 0.15:
        return rand_json(rng)
    obj = {}
    if rng.random() < 0.9:
        bad = ["x" * 300, "", "a\ud800", 7, None, rand_text(rng)]
        obj["id"] = f"row-{rng.randint(0, 99)}" if rng.random() < 0.7 else rng.choice(bad)
    if rng.random() < 0.9:
        if rng.random() < 0.85:
            obj["context"] = [
                {"speaker": rng.choice(["customer", "agent", "action", "AGENT", "", None, 3]),
                 "text": rand_text(rng) if rng.random() < 0.85 else rand_json(rng)}
                if rng.random() < 0.9 else rand_json(rng) for _ in range(rng.randint(0, 12))]
        else:
            obj["context"] = rand_json(rng)
    if rng.random() < 0.6:
        obj["turn_index"] = rng.choice([0, 5, -1, 10**30, 2.5, True, "3", None, 9999, 10000])
    if rng.random() < 0.2:
        obj["intent"] = rand_json(rng)                            # gold-looking keys must be ignored
    return obj


def test_fuzz_never_raises(triage_off):
    rng = random.Random(0)
    objs = [rand_obj(rng) for _ in range(500)]
    single = [triage_off.handle(o, f"f{i}") for i, o in enumerate(objs)]
    batch = triage_off.handle_many(objs, [f"f{i}" for i in range(len(objs))])
    answered = 0
    for o, (resp, tr), (bresp, btr) in zip(objs, single, batch):
        assert schema_valid(resp, None), resp
        json.dumps(resp), json.dumps(tr)
        assert resp["id"] == (o["id"] if gate.envelope_error(o) is None else "")
        assert tr["error"] is None, tr                            # garbage abstains; it is never an internal error
        assert (bresp, same_trace(btr, tr)) == (resp, True)
        answered += "error" not in resp
    assert answered >= 100                                        # the model path ran, not only abstentions


def test_internal_error_envelope(triage_off, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model broke")
    monkeypatch.setattr(triage_off.cpu, "raw_probs", boom)
    good = {"id": "g-1", "context": [{"speaker": "customer", "text": "where is my order"},
                                     {"speaker": "agent", "text": "let me check"}]}
    empty = {"id": "e-1", "context": [{"speaker": "customer", "text": ""}]}
    resp, tr = triage_off.handle(good, "r")
    assert schema_valid(resp, None) and resp["needs_human"] is True
    assert resp["error"] == {"code": "internal", "message": "RuntimeError"} and resp["id"] == "g-1"
    assert tr["error"] == "RuntimeError" and "model broke" not in json.dumps(tr)
    out = triage_off.handle_many([good, empty])
    assert out[0][0]["error"]["code"] == "internal" and out[1][0]["error"]["code"] == "invalid_input"


# ----------------------------------------------------------------------------- batch against single calls

def test_handle_many_matches_handle(triage_off, split):
    objs = [project(r) for r in split("dev")[:240] + split("noisy")[:51] + split("hostile")]
    assert len(objs) == 300
    batch = triage_off.handle_many(objs, [f"b{i}" for i in range(300)])
    for i, o in enumerate(objs):
        resp, tr = triage_off.handle(o, f"b{i}")
        assert batch[i][0] == resp, o["id"]
        assert same_trace(batch[i][1], tr), o["id"]


# ----------------------------------------------------------------------------- the LLM step (fake)

def test_llm_agree_raises_confidence(triage_off, soft, monkeypatch):
    row, cpu, _, _ = soft
    fake = FakeStep(LLMResult("ok", intent="x", action=cpu["action"], latency_s=0.5, cost_usd=0.0001,
                              model="m", provider="p"))
    monkeypatch.setattr(triage_off, "llm", fake)
    resp, tr = triage_off.handle(row, "r")
    assert fake.calls == [row["id"]]
    assert resp["action"] == cpu["action"] and resp["intent"] == cpu["intent"]   # the LLM's intent is not used
    assert resp["confidence"] == round((cpu["confidence"] + 1.0) / 2, 6)
    assert resp["needs_human"] is True and tr["route_reasons"] == ["low_confidence"]   # still under 0.8498
    assert tr["llm"] == {"status": "ok", "action": cpu["action"], "used": True, "latency_s": 0.5,
                         "cost_usd": 0.0001, "model": "m", "provider": "p"}
    assert tr["cpu"]["confidence"] == cpu["confidence"]


def test_llm_disagree_takes_llm_action(triage_off, soft, monkeypatch):
    row, cpu, _, other = soft
    monkeypatch.setattr(triage_off, "llm", FakeStep(LLMResult("ok", action=other)))
    resp, tr = triage_off.handle(row, "r")
    assert resp["action"] == other and resp["intent"] == cpu["intent"]
    assert resp["confidence"] == round(cpu["confidence"] / 2, 6)
    assert tr["cpu"]["action"] == cpu["action"] and "off_policy" not in tr["flags"]


@pytest.mark.parametrize("status", ["timeout", "http_error", "invalid", "abstained", "breaker_open", "disabled"])
def test_llm_failure_keeps_cpu_answer(triage_off, soft, monkeypatch, status):
    row, cpu, _, other = soft
    monkeypatch.setattr(triage_off, "llm", FakeStep(LLMResult(status, action=other, latency_s=4.0)))
    resp, tr = triage_off.handle(row, "r")
    assert resp == cpu
    assert tr["llm"]["status"] == status and tr["llm"]["used"] is False


def test_llm_exception_keeps_cpu_answer(triage_off, soft, monkeypatch):
    row, cpu, _, _ = soft
    monkeypatch.setattr(triage_off, "llm", FakeStep(RuntimeError("bug")))
    resp, tr = triage_off.handle(row, "r")
    assert resp == cpu and tr["llm"]["status"] == "invalid" and tr["error"] is None


def test_llm_action_outside_labels_ignored(triage_off, soft, monkeypatch):
    row, cpu, _, _ = soft
    monkeypatch.setattr(triage_off, "llm", FakeStep(LLMResult("ok", action="refund-everything")))
    resp, tr = triage_off.handle(row, "r")
    assert resp == cpu and tr["llm"]["used"] is False


def test_masked_llm_action_ignored(triage_off, soft, monkeypatch, tmp_path):
    row, cpu, tr0, other = soft
    need = unseen(row, other)
    store = write_policy(tmp_path / "p.toml", rule("no-other", other, need, "mask"))
    monkeypatch.setattr(triage_off, "store", store)
    base, btr = triage_off.handle(row, "r")                       # mask policy, LLM off
    assert other in btr["cpu"]["masked"] and base["action"] == cpu["action"]
    fake = FakeStep(LLMResult("ok", action=other))
    monkeypatch.setattr(triage_off, "llm", fake)
    resp, tr = triage_off.handle(row, "r")
    assert fake.calls and resp == base
    assert tr["llm"]["used"] is False and tr["policy_version"] == "t.1"


def test_flagged_and_confident_rows_never_call_llm(triage_off, split, dev_out, labelled, monkeypatch):
    fake = FakeStep(LLMResult("ok", action="search-faq"), gate=1.0)
    monkeypatch.setattr(triage_off, "llm", fake)
    hostile = [project(r) for r in split("hostile")]
    for o in hostile:
        triage_off.handle(o, "r")
    triage_off.handle_many(hostile)
    flagged = [project(r) for r, (_, tr) in zip(labelled, dev_out) if set(tr["flags"]) & gate.ROUTE_FLAGS]
    assert flagged                                                # the no_dialogue dev row and its noisy twin
    triage_off.handle_many(flagged)
    assert fake.calls == []
    fake.cfg = LLMConfig(mode="on", gate=0.0)                     # nothing is below a gate of 0
    triage_off.handle_many([project(r) for r in labelled[:50]])
    assert fake.calls == []


def test_llm_gate_is_strict(triage_off, soft, monkeypatch):
    row, cpu, _, _ = soft
    fake = FakeStep(LLMResult("timeout"), gate=cpu["confidence"])
    monkeypatch.setattr(triage_off, "llm", fake)
    triage_off.handle(row, "r")
    assert fake.calls == []
    fake.cfg = LLMConfig(mode="on", gate=cpu["confidence"] + 1e-6)
    triage_off.handle(row, "r")
    assert fake.calls == [row["id"]]


def test_handle_many_with_llm_matches_handle(triage_off, labelled, monkeypatch):
    fake = FakeStep(LLMResult("ok", action="pull-up-account"), gate=0.7)
    monkeypatch.setattr(triage_off, "llm", fake)
    objs = [project(r) for r in labelled[:60]]
    batch = triage_off.handle_many(objs, [str(i) for i in range(60)])
    asked = len(fake.calls)
    assert asked > 5                                              # the pool of 8 ran
    for i, o in enumerate(objs):
        resp, tr = triage_off.handle(o, str(i))
        assert batch[i][0] == resp and same_trace(batch[i][1], tr)
    assert len(fake.calls) == 2 * asked


# ----------------------------------------------------------------------------- policy wiring and routing

def test_prior_follows_policy_file_and_reload(triage_off, labelled, monkeypatch, tmp_path):
    objs = [project(r) for r in labelled[:20]]
    path = tmp_path / "p.toml"
    store = write_policy(path, flows={f: ["pull-up-account"] for f in sorted(triage_off.cpu.fcls)})
    monkeypatch.setattr(triage_off, "store", store)
    out = triage_off.handle_many(objs)
    assert {resp["action"] for resp, _ in out} == {"pull-up-account"}
    assert all(tr["policy_version"] == "t.1" for _, tr in out)
    write_policy(path, version="t.2")                             # back to the shipped lists; picked up live
    out2 = triage_off.handle_many(objs)
    assert len({resp["action"] for resp, _ in out2}) > 1 and out2[0][1]["policy_version"] == "t.2"
    assert triage_off.health()["policy"]["version"] == "t.2"


def test_mask_rule_removes_cpu_action(triage_off, soft, monkeypatch, tmp_path):
    row, cpu, tr0, _ = soft
    need = unseen(row, cpu["action"])
    monkeypatch.setattr(triage_off, "store", write_policy(tmp_path / "p.toml", rule("m", cpu["action"], need, "mask")))
    resp, tr = triage_off.handle(row, "r")
    assert resp["action"] != cpu["action"] and tr["cpu"]["masked"] == [cpu["action"]]
    assert "all_masked" not in tr["flags"] and not any(f.startswith("rule:") for f in tr["flags"])


def test_all_masked_keeps_answer_and_routes(triage_off, soft, monkeypatch, tmp_path):
    row, *_ = soft
    store = write_policy(tmp_path / "p.toml", rule("only-after-reset", "pull-up-account", "make-password", "mask"),
                         flows={f: ["pull-up-account"] for f in sorted(triage_off.cpu.fcls)})
    monkeypatch.setattr(triage_off, "store", store)
    resp, tr = triage_off.handle(row, "r")
    assert resp["action"] == "pull-up-account"
    assert {"all_masked", "rule:only-after-reset"} <= set(tr["flags"])
    assert resp["needs_human"] is True and resp["confidence"] <= 0.5
    assert "rule:only-after-reset" in tr["route_reasons"]


def test_route_rule_routes_and_caps(triage_off, dev_out, labelled, monkeypatch, tmp_path):
    k = next(i for i, (resp, tr) in enumerate(dev_out) if not tr["route_reasons"] and resp["confidence"] > 0.5)
    row, cpu = project(labelled[k]), dev_out[k][0]
    need = unseen(row, cpu["action"])
    monkeypatch.setattr(triage_off, "store", write_policy(tmp_path / "p.toml", rule("r1", cpu["action"], need, "route")))
    resp, tr = triage_off.handle(row, "r")
    assert resp["action"] == cpu["action"] and resp["needs_human"] is True and resp["confidence"] == 0.5
    assert tr["route_reasons"] == ["rule:r1"] and tr["cpu"]["masked"] == []


def test_off_policy_routing_switch(triage_off, dev_out, labelled, v0, monkeypatch):
    k = next(i for i, (resp, tr) in enumerate(dev_out) if tr["route_reasons"] == ["off_policy"])
    row, ref = project(labelled[k]), v0[labelled[k]["id"]]
    resp, _ = triage_off.handle(row, "r")
    assert resp["needs_human"] is True and resp["confidence"] == 0.5 < ref["confidence"]
    monkeypatch.setattr(triage_off, "settings", Settings(route_off_policy=False))
    resp, tr = triage_off.handle(row, "r")
    assert {**resp, "confidence": 0} == {**ref, "confidence": 0} and tr["route_reasons"] == []
    assert abs(resp["confidence"] - ref["confidence"]) <= CONF_TOL and "off_policy" in tr["flags"]


def test_second_intent_flag_and_switch(triage_off, dev_out, labelled, monkeypatch):
    for _, tr in dev_out:
        second = tr["cpu"]["second_intent"]
        fires = second is not None and second[1] >= 0.6 and triage_off.intent_flow[second[0]] != tr["cpu"]["flow"]
        assert ("second_intent" in tr["flags"]) == fires
    k = next(i for i, (_, tr) in enumerate(dev_out) if "second_intent" in tr["flags"] and not tr["route_reasons"])
    row, (resp0, tr0) = project(labelled[k]), dev_out[k]
    second, p = tr0["cpu"]["second_intent"]
    assert p >= 0.6 and triage_off.intent_flow[second] != tr0["cpu"]["flow"]
    assert resp0["needs_human"] is False                           # a flag only, by default
    monkeypatch.setattr(triage_off, "settings", Settings(route_second_intent=True))
    resp, tr = triage_off.handle(row, "r")
    assert resp["needs_human"] is True and tr["route_reasons"] == ["second_intent"] and resp["confidence"] <= 0.5
    monkeypatch.setattr(triage_off, "settings", Settings(second_intent_p=1.0))
    assert "second_intent" not in triage_off.handle(row, "r")[1]["flags"]


# ----------------------------------------------------------------------------- set-up and health

def test_health(triage_off):
    h = triage_off.health()
    assert h["status"] == "ok" and h["bundle"] == "models/classical_v0.joblib"
    assert h["bundle_sha256"] == hashlib.sha256(triage_off.bundle_path.read_bytes()).hexdigest()
    assert h["policy"] == {"version": "2026-10-14.1", "path": "policy/policy.toml", "last_error": None,
                           "loaded_at": h["policy"]["loaded_at"]}
    assert h["llm"]["enabled"] is False and h["llm"]["breaker_open"] is False
    assert h["llm"]["base_url_host"] == "openrouter.ai" and h["started_at"]
    json.dumps(h)


def test_intent_flow_map(triage_off):
    m = triage_off.intent_flow
    assert set(map(str, triage_off.cpu.icls)) <= set(m) and set(m.values()) == set(map(str, triage_off.cpu.fcls))
    assert intent_flows({"intent_flow": {"a": "b"}}) == {"a": "b"}


def test_settings_from_env():
    assert Settings.from_env({}) == Settings(0.6, False, True, 0.5)
    s = Settings.from_env({"TRIAGE_SECOND_INTENT_P": "0.7", "TRIAGE_ROUTE_SECOND_INTENT": "1",
                           "TRIAGE_ROUTE_OFF_POLICY": "0", "TRIAGE_FLAG_CONF_CAP": "0.4"})
    assert s == Settings(0.7, True, False, 0.4)
    for k, v in [("TRIAGE_FLAG_CONF_CAP", "2"), ("TRIAGE_SECOND_INTENT_P", "x"), ("TRIAGE_ROUTE_OFF_POLICY", "maybe")]:
        with pytest.raises(ValueError, match=k):
            Settings.from_env({k: v})


def test_llm_off_or_missing_index(tmp_path):
    assert load_llm(LLMConfig(mode="off")) == (None, None)
    assert load_llm(LLMConfig(mode="auto")) == (None, None)       # auto with no key stays off
    step, err = load_llm(LLMConfig(mode="on"), tmp_path / "missing.joblib")
    assert step is None and "build_retrieval" in err


def test_from_env_reads_paths(monkeypatch):
    seen = {}
    monkeypatch.setattr(Triage, "__init__", lambda self, **kw: seen.update(kw))
    Triage.from_env({"TRIAGE_BUNDLE": "b.joblib", "TRIAGE_POLICY_PATH": "p.toml", "TRIAGE_LLM": "off",
                     "TRIAGE_FLAG_CONF_CAP": "0.3"})
    assert str(seen["bundle"]) == "b.joblib" and str(seen["policy_path"]) == "p.toml"
    assert seen["llm_cfg"].enabled is False and seen["settings"].flag_conf_cap == 0.3
