"""scripts/reproduce.sh: bad settings stop `serve` before a server starts, and the eval commands rebuild the
manifest. Runs the script for real; no server comes up."""
from __future__ import annotations

import os
import re
import socket
import subprocess

import pytest

from triage.data import ROOT

SCRIPT = ROOT / "scripts" / "reproduce.sh"


def _function(name: str) -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    return re.search(rf"^{name}\(\) \{{\n(.*?)^\}}", text, re.M | re.S).group(1)


@pytest.mark.parametrize("name", ["cmd_eval", "cmd_http_eval"])
def test_eval_commands_rebuild_the_manifest(name):
    assert re.search(r"^\s*manifest\s*$", _function(name), re.M)
    assert "harness.manifest" in _function("manifest")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("setting, want", [
    ({"TRIAGE_POLICY_PATH": "BAD"}, "version must be set"),
    ({"TRIAGE_ROUTE_OFF_POLICY": "maybe"}, "TRIAGE_ROUTE_OFF_POLICY must be 1 or 0"),
    ({"TRIAGE_LLM": "maybe"}, "TRIAGE_LLM must be on, off or auto"),
])
def test_serve_stops_on_a_bad_setting(tmp_path, setting, want):
    """A bad setting or policy file ends `serve` at once with the reason, instead of a process serving 503."""
    bad = tmp_path / "policy.toml"
    bad.write_text("version = 3\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if "API_KEY" not in k and not k.startswith("TRIAGE_")}
    env.update({"TRIAGE_LLM": "off", "PORT": str(_free_port()), "HOST": "127.0.0.1"})
    env.update({k: str(bad) if v == "BAD" else v for k, v in setting.items()})
    p = subprocess.run(["bash", str(SCRIPT), "serve"], cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=60)
    assert p.returncode != 0 and want in p.stderr, p.stderr[-2000:]
    assert "starting on" not in p.stderr
