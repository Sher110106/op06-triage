"""Replay dev + dev_noisy under the two refund-rule policies in this folder and save what scripts/replay_policy.py
reports, so the numbers DECISIONS.md D1 and RUNBOOK.md quote have files behind them.

  .venv/bin/python experiments/policy_replay/run.py

For each effect ("route", "mask") it writes results/raw/policy-replay-refund-<effect>/:
  report.json  scripts/replay_policy.py --out
  report.txt   the text the same command printed
  meta.json    command, policy files and their SHA-256, model bundle, input data, machine, times

The LLM step is off in both passes (replay_policy.py sets TRIAGE_LLM=off), and every TRIAGE_* setting and API key
is removed from the environment, so the shipped defaults apply and no call leaves the machine.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CURRENT = "policy/policy.toml"
BUNDLE = "models/classical_v0.joblib"
DATA = {"dev": "data/op06/dev.jsonl", "dev_noisy": "data/op06/dev_noisy.jsonl", "hostile": "data/op06/hostile.jsonl"}
KEYS = ("OPENROUTER_API_KEY", "TRIAGE_LLM_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def main() -> int:
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRIAGE_") and k not in KEYS}
    for effect in ("route", "mask"):
        new = f"experiments/policy_replay/policy_refund_{effect}.toml"
        out = ROOT / "results" / "raw" / f"policy-replay-refund-{effect}"
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "scripts/replay_policy.py", "--current", CURRENT, "--policy", new,
               "--out", str((out / "report.json").relative_to(ROOT))]
        started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        t0 = time.time()
        p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        if p.returncode != 0:
            sys.stderr.write(p.stderr)
            return p.returncode
        (out / "report.txt").write_text(p.stdout if p.stdout.endswith("\n") else p.stdout + "\n", encoding="utf-8")
        rep = json.loads((out / "report.json").read_text())
        meta = {
            "run_id": out.name, "kind": "policy_replay", "created": started,
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(time.time() - t0, 1),
            "script": "experiments/policy_replay/run.py",
            "command": " ".join([".venv/bin/python", *cmd[1:]]),
            "what": f"dev + dev_noisy replayed under the shipped policy and under a copy with the refund rule "
                    f"(offer-refund needs verify-identity or validate-purchase earlier), effect = {effect!r}",
            "llm": "off (scripts/replay_policy.py sets TRIAGE_LLM=off); no API call",
            "environment": "every TRIAGE_* variable and API key removed, so the shipped defaults apply",
            "policy_current": {"file": CURRENT, "sha256": sha(CURRENT), "version": rep["current"]["version"]},
            "policy_new": {"file": new, "sha256": sha(new), "version": rep["new"]["version"]},
            "bundle_sha256": sha(BUNDLE),
            "inputs": {k: {"file": f, "sha256": sha(f)} for k, f in DATA.items()},
            "machine": {"platform": platform.platform(), "python": platform.python_version()},
            "outputs": ["report.json", "report.txt"],
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(f"wrote {out.relative_to(ROOT)}: {rep['changes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
