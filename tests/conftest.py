"""Shared test setup: the repo root on sys.path, and data splits read once per test session."""
from __future__ import annotations

import sys
from functools import cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def split():
    """split("dev") -> the rows of that split ("train", "dev", "noisy", "hostile"), loaded once. Do not mutate."""
    from triage.data import load_split
    return cache(load_split)


@pytest.fixture(scope="session")
def triage_off():
    """triage.pipeline.Triage with the LLM step off whatever the shell sets; bundle and shipped policy loaded once.
    Tests that change its attributes must use monkeypatch so the next test sees the original."""
    from triage.classical import DEFAULT_BUNDLE
    if not DEFAULT_BUNDLE.exists():
        pytest.skip("models/classical_v0.joblib is not built; run scripts/reproduce.sh train")
    from triage.pipeline import Triage
    tri = Triage.from_env({"TRIAGE_LLM": "off"})
    yield tri
    tri.close()
