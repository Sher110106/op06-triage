"""Throughput of the in-process pipeline (LLM step off) by thread count, to record the P3-2 finding.

  python experiments/linux_threads/scaling.py --out results/raw/linux-threads/scaling.json

Three modes: "dedicated" (the shipped design: CPU stages on one dedicated thread), "caller" (the old design:
each request's thread runs the CPU stages itself; Triage._on_cpu is patched to call the stage directly), and
"caller-pinned" (the old design with the process pinned to one core, Linux only). 800 dev rows per point,
30 warm-up rows, BLAS threads set to 1 by the caller's environment.
"""
import argparse
import concurrent.futures as cf
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triage.data import load_split, project  # noqa: E402
from triage.pipeline import Triage  # noqa: E402


def run(t, rows, n):
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(n) as ex:
        list(ex.map(lambda r: t.handle(r), rows))
    return round(len(rows) / (time.perf_counter() - t0), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=800)
    a = ap.parse_args()
    rows = [project(r) for r in load_split("dev")[: a.rows]]
    res = {"rows": len(rows), "platform": platform.platform(), "python": platform.python_version(),
           "cpu_count": os.cpu_count(), "blas_threads_env": os.environ.get("OPENBLAS_NUM_THREADS"),
           "load_avg_start": os.getloadavg(), "modes": {}}
    for mode in ("dedicated", "caller", "caller-pinned"):
        if mode == "caller-pinned":
            if not hasattr(os, "sched_setaffinity"):
                continue
            before = os.sched_getaffinity(0)
            os.sched_setaffinity(0, {min(before)})
        t = Triage()
        if mode.startswith("caller"):
            t._on_cpu = lambda fn, *args: fn(*args)
        for r in rows[:30]:
            t.handle(r)
        res["modes"][mode] = {str(n): run(t, rows, n) for n in (1, 2, 4, 8)}
        t.close()
        if mode == "caller-pinned":
            os.sched_setaffinity(0, before)
        print(mode, res["modes"][mode], flush=True)
    res["load_avg_end"] = os.getloadavg()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
