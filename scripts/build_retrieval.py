"""Build the retrieval index for the LLM action step over all train rows and save it.

  .venv/bin/python scripts/build_retrieval.py [--out models/retrieval.joblib]
  .venv/bin/python scripts/build_retrieval.py --bench [--rows 200]   # load the saved file, time top_k on dev rows

Prints one JSON line. Run --bench in its own process so the RSS figure covers the load alone.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage.data import load_split, project  # noqa: E402
from triage.retrieval import DEFAULT_PATH, Retriever  # noqa: E402


def rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss                 # bytes on macOS, KiB on Linux
    return round(peak / (1 << 20) if sys.platform == "darwin" else peak / 1024, 1)


def build(out: Path) -> dict:
    t0 = time.time()
    rows = load_split("train")
    r = Retriever.build(rows)
    t1 = time.time()
    r.save(out)
    return {"rows": len(rows), "features": r.by_feature.shape[0], "nnz": int(r.by_feature.nnz),
            "build_s": round(t1 - t0, 1), "save_s": round(time.time() - t1, 1),
            "file_mb": round(os.path.getsize(out) / 1e6, 1), "path": str(out)}


def bench(path: Path, n: int) -> dict:
    t0 = time.time()
    r = Retriever.load(path)
    load_s = time.time() - t0
    rss = rss_mb()
    rows = [project(x) for x in load_split("dev")[:n]]
    lat = []
    for row in rows:
        t = time.perf_counter()
        r.examples(row)
        lat.append(time.perf_counter() - t)
    lat.sort()
    return {"load_s": round(load_s, 2), "rss_mb_after_load": rss, "rows": len(rows),
            "top_k_ms_mean": round(1000 * sum(lat) / len(lat), 2), "top_k_ms_p50": round(1000 * lat[len(lat) // 2], 2),
            "top_k_ms_p95": round(1000 * lat[int(0.95 * (len(lat) - 1))], 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--rows", type=int, default=200)
    a = ap.parse_args()
    print(json.dumps(bench(a.out, a.rows) if a.bench else build(a.out)), flush=True)
