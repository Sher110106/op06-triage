"""docs/DESIGN.md against the code and the data: settings, run command, parity exceptions and flag counts."""
from __future__ import annotations

import re

from triage import gate
from triage.data import ROOT, project

DESIGN = (ROOT / "docs" / "DESIGN.md").read_text(encoding="utf-8")


def _env_table() -> set[str]:
    return set(re.findall(r"^\| `(TRIAGE_[A-Z_]+)` \|", DESIGN, re.M))


def test_every_setting_is_in_the_env_table():
    """README.md says docs/DESIGN.md lists every setting."""
    used = set()
    for path in [*(ROOT / "triage").glob("*.py"), *(ROOT / "scripts").glob("*.py")]:
        used |= set(re.findall(r"\b(TRIAGE_[A-Z_]+)\b", path.read_text(encoding="utf-8")))
    assert used and used <= _env_table(), sorted(used - _env_table())


def test_run_command_matches_the_launcher():
    run = re.search(r"Run: `uvicorn triage\.service:app (.+?)`", DESIGN).group(1).split()
    launcher = (ROOT / "scripts" / "reproduce.sh").read_text(encoding="utf-8")
    serve_args = re.search(r"^SERVE_ARGS=\((.+)\)$", launcher, re.M).group(1).split()
    assert run[run.index("--port") + 2:] == serve_args
    assert "--no-access-log" in run


def test_request_flow_and_parity_name_every_default_route():
    para = DESIGN.split("## Request flow", 1)[1].split("## Modules and interfaces", 1)[0]
    assert "gate ROUTE flag" in para and "off_policy" in para and "TRIAGE_ROUTE_OFF_POLICY" in para
    assert "policy_uncertainty" in para and "llm_disagreement" in para


def test_route_flags_listed():
    listed = re.search(r"`ROUTE_FLAGS = \{(.+?)\}`", DESIGN).group(1)
    assert {x.strip() for x in listed.split(",")} == gate.ROUTE_FLAGS


def test_no_dialogue_counts(split):
    def count(rows):
        return sum(all(t["speaker"] == "customer" for t in project(r)["context"]) for r in rows)

    eval_n, train_n = count(split("dev") + split("noisy")), count(split("train"))
    line = DESIGN[DESIGN.index("- `no_dialogue`"):].split("\n- ")[0]
    assert f"{eval_n} of 2,300 dev + dev_noisy rows" in line and f"{train_n} of 20,000 train rows" in line
