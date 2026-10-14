"""Shared test setup: the repo root on sys.path, data splits read once per test session, and the Triage objects.

`triage_off` routes as the v0 model did (both routing points at the bundle's route_threshold), so tests can check
parity with results/raw/dev-classical-v0. `triage_shipped` uses policy/policy.toml as shipped (Phase 2 points).
"""
from __future__ import annotations

import importlib.util
import re
import sys
import threading
from functools import cache
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@cache
def fake_upstream_module():
    """experiments/chaos/fake_upstream.py, imported once."""
    spec = importlib.util.spec_from_file_location("op06_fake_upstream", ROOT / "experiments/chaos/fake_upstream.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_upstream():
    """start(mode) -> (base URL ending in /v1, its State): experiments/chaos/fake_upstream.py on 127.0.0.1 in this
    process, answering at once (no delay), no replay file. Stopped after the test."""
    servers = []

    def start(mode: str):
        fu = fake_upstream_module()
        state = fu.State(mode, 0.0, 0.0, 0.0, 0, {}, {"ambiguous": set()})
        srv = ThreadingHTTPServer(("127.0.0.1", 0), fu.make_handler(state))
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}/v1", state

    yield start
    for srv in servers:
        srv.shutdown()
        srv.server_close()

V0_ROUTE_THRESHOLD = 0.8498168103175511     # route_threshold of models/classical_v0.joblib, written in full


def v0_policy_text(version: str | None = None) -> str:
    """policy/policy.toml with both routing points set to v0's and the version changed (default: the shipped
    version plus "-v0"). Everything else is the shipped file."""
    from triage.policy import DEFAULT_POLICY, loads
    text = DEFAULT_POLICY.read_text(encoding="utf-8")
    shipped = loads(text)
    head, sep, tail = text.partition("\n[thresholds]\n")
    tail, n = re.subn(r"(?m)^(high|low) = [0-9.]+$", lambda m: f"{m[1]} = {V0_ROUTE_THRESHOLD!r}", tail)
    assert sep and n == 2, "policy/policy.toml has no [thresholds] section with high and low"
    head, n = re.subn(r'(?m)^version = ".*"$', f'version = "{version or shipped.version + "-v0"}"', head)
    assert n == 1, "policy/policy.toml has no version line"
    return head + sep + tail


@pytest.fixture(scope="session")
def split():
    """split("dev") -> the rows of that split ("train", "dev", "noisy", "hostile"), loaded once. Do not mutate."""
    from triage.data import load_split
    return cache(load_split)


@pytest.fixture(scope="session")
def v0_policy(tmp_path_factory) -> Path:
    """A copy of the shipped policy that routes as v0 did: see v0_policy_text."""
    path = tmp_path_factory.mktemp("policy") / "policy-v0.toml"
    path.write_text(v0_policy_text(), encoding="utf-8")
    return path


def _triage(policy: Path):
    from triage.classical import DEFAULT_BUNDLE
    if not DEFAULT_BUNDLE.exists():
        pytest.skip("models/classical_v0.joblib is not built; run scripts/reproduce.sh train")
    from triage.pipeline import Triage
    return Triage.from_env({"TRIAGE_LLM": "off", "TRIAGE_POLICY_PATH": str(policy)})


@pytest.fixture(scope="session")
def triage_off(v0_policy):
    """triage.pipeline.Triage with the LLM step off whatever the shell sets, and the v0 routing points (v0_policy),
    so the CPU path can be checked against dev-classical-v0. Bundle and policy loaded once.
    Tests that change its attributes must use monkeypatch so the next test sees the original."""
    tri = _triage(v0_policy)
    yield tri
    tri.close()


@pytest.fixture(scope="session")
def triage_shipped():
    """Like triage_off, but with policy/policy.toml as shipped: what `scripts/reproduce.sh serve` runs."""
    from triage.policy import DEFAULT_POLICY
    tri = _triage(DEFAULT_POLICY)
    yield tri
    tri.close()
