"""Re-run every analysis that reads the moved raw files and compare its outputs with a saved copy.

The raw LLM call logs, the shared call ledger and the encoder cache moved from experiments/ to results/raw/
(move_raw.py; old -> new paths in moves.json). This script proves the move changed no result.

  python experiments/relocate/rerun_check.py snapshot NAME     copy every output file below to WORK/NAME/
  python experiments/relocate/rerun_check.py rerun NAME        run every analysis, then copy its outputs to WORK/NAME/
  python experiments/relocate/rerun_check.py compare A B [--out FILE]   compare WORK/A with WORK/B
  python experiments/relocate/rerun_check.py restore NAME [--rewrite-paths]   copy WORK/NAME back over the repo

WORK is results/scratch/relocate/ (git-ignored). The analyses run with whatever raw paths exist: the old ones
before the move, the new ones after it. No analysis here calls an API.

The comparison is byte for byte first. A file that differs is parsed (JSON, JSONL, gzipped JSONL) and compared
field by field. Two kinds of difference are allowed, and both are counted and reported:
  - wall-clock fields (WALL_CLOCK): when a run happened, how long it took, the machine's load, random request ids;
  - path strings: the old raw-file paths are rewritten to the new ones (PATHS) in both copies before comparing.
Any other difference is listed in full, with the explanation from EXPLAINED if there is one.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = ROOT / "results" / "scratch" / "relocate"
PY = sys.executable

# old raw path -> new raw path (the strings recorded inside result and meta files)
PATHS = {
    "experiments/llm_tail/runs/luna_V3_none.jsonl": "results/raw/llm-tail-luna_V3_none/raw_calls.jsonl",
    "experiments/llm_tail/runs/haiku_V3_low.jsonl": "results/raw/llm-tail-haiku_V3_low/raw_calls.jsonl",
    "experiments/llm_confirm/runs/luna_B.jsonl": "results/raw/llm-confirm-luna_B/raw_calls.jsonl",
    "experiments/llm_confirm/runs/haiku_B.jsonl": "results/raw/llm-confirm-haiku_B/raw_calls.jsonl",
}

# fields that hold wall-clock facts, not results
WALL_CLOCK = {
    "created": "time the run was made",
    "batch_wall_s": "seconds the batch took",
    "timings_ms": "per-request stage timings",
    "request_id": "random id per request (uuid4)",
    "load_avg": "machine load average when the run started",
    "loaded_at": "time the policy file was loaded",
}

# differences that are not wall-clock and not the move, with the reason (filled in after reading the diffs)
EXPLAINED = {
    ("results/raw/dev-pipeline-llm-replay/traces.jsonl", "policy_version"):
        "policy/policy.toml was saved as version 2026-10-14.2 at 16:28 on 14 Oct, five minutes after the committed "
        "run (16:23, version 2026-10-14.1). Every other field of all 2,309 trace lines is equal, and the predictions "
        "are byte-identical, so the policy edit changed no decision on these rows.",
    ("experiments/llm_confirm/results_B.json", "action_step_cost_latency"):
        "analyse_B.py does not write this block; the committed file got it from a step that was not saved. Found "
        "before the move. experiments/llm_confirm/cost_latency_B.py (new) rebuilds it from the raw files, and "
        "analyse_B.py followed by it gives the committed file byte for byte (the 'final' re-run).",
}


def raw_inputs() -> dict:
    """The recorded LLM answers and the encoder cache, at the new location if it exists, else the old one."""
    new = {"luna_dev": ROOT / "results/raw/llm-tail-luna_V3_none/raw_calls.jsonl",
           "haiku_dev": ROOT / "results/raw/llm-tail-haiku_V3_low/raw_calls.jsonl",
           "luna_B": ROOT / "results/raw/llm-confirm-luna_B/raw_calls.jsonl",
           "haiku_B": ROOT / "results/raw/llm-confirm-haiku_B/raw_calls.jsonl",
           "encoder": ROOT / "results/raw/encoder-ett32A"}
    old = {"luna_dev": ROOT / "experiments/llm_tail/runs/luna_V3_none.jsonl",
           "haiku_dev": ROOT / "experiments/llm_tail/runs/haiku_V3_low.jsonl",
           "luna_B": ROOT / "experiments/llm_confirm/runs/luna_B.jsonl",
           "haiku_B": ROOT / "experiments/llm_confirm/runs/haiku_B.jsonl",
           "encoder": ROOT / "experiments/encoder/ett32A"}
    moved = all(p.exists() for p in new.values())
    if not moved:
        assert all(p.exists() for p in old.values()), "raw inputs are at neither the old nor the new paths"
    return {"layout": "new" if moved else "old", **(new if moved else old)}


def analyses(inp: dict) -> list[tuple[str, list[str], list[str]]]:
    """(name, command, output files relative to ROOT), in the order they must run."""
    abl = [f"results/raw/ablation-{n}/{f}" for n in ("ref-llm", "llm-10", "llm-30")
           for f in ("predictions.jsonl", "traces.jsonl.gz", "meta.json", "report.json", "report_official.json",
                     "ablation.json")]
    tail = [f"results/raw/llm-tail-{s}/{f}" for s in ("A_luna_tail", "B_haiku_intent", "C_both")
            for f in ("predictions.jsonl", "report.json", "report_official.json")]
    replay = [f"results/raw/dev-pipeline-llm-replay/{f}" for f in
              ("predictions.jsonl", "traces.jsonl", "meta.json", "report.json", "report_official.json")]
    chaos = [f"results/raw/chaos-ref-replay/{f}" for f in
             ("predictions.jsonl", "traces.jsonl", "meta.json", "report.json", "report_official.json")]
    new = inp["layout"] == "new"       # after the move, run with the scripts' own default paths (the new ones)
    return [
        ("llm_tail/analyse.py",
         [PY, "experiments/llm_tail/analyse.py"] + ([] if new else [
             "--cpu", "results/raw/dev-classical-v0/predictions.jsonl",
             "--luna", str(inp["luna_dev"]), "--haiku", str(inp["haiku_dev"])]),
         ["experiments/llm_tail/results.json", *tail]),
        ("llm_confirm/analyse_B.py",
         [PY, "experiments/llm_confirm/analyse_B.py"] + ([] if new else [
             "--luna", str(inp["luna_B"]), "--haiku", str(inp["haiku_B"])]),
         ["experiments/llm_confirm/results_B.json"]),
        *([("llm_confirm/cost_latency_B.py", [PY, "experiments/llm_confirm/cost_latency_B.py"],
            ["experiments/llm_confirm/results_B.json"])] if new else []),
        ("encoder/analyse_encoder.py",
         [PY, "experiments/encoder/analyse_encoder.py"] + ([] if new else ["--cache", str(inp["encoder"])]),
         ["experiments/encoder/results_encoder.json"]),
        ("llm_confirm/choose_rule.py", [PY, "experiments/llm_confirm/choose_rule.py"],
         ["experiments/llm_confirm/rule_selection_dev.json"]),
        ("routing/analyse_routing.py", [PY, "experiments/routing/analyse_routing.py"],
         ["experiments/routing/results_routing.json", *replay]),
        ("ablations/run.py ref-llm llm-10 llm-30",
         [PY, "experiments/ablations/run.py", "ref-llm", "llm-10", "llm-30"], abl),
        ("ablations/run.py llm-share-fold-b", [PY, "experiments/ablations/run.py", "llm-share-fold-b"],
         ["experiments/ablations/llm_share_fold_b.json"]),
        ("ablations/run.py table", [PY, "experiments/ablations/run.py", "table"],
         ["experiments/ablations/ablations.json"]),
        ("chaos/run_chaos.py ref-replay", [PY, "experiments/chaos/run_chaos.py", "ref-replay"], chaos),
        ("chaos/run_chaos.py report", [PY, "experiments/chaos/run_chaos.py", "report"],
         ["experiments/chaos/chaos.json"]),
    ]


def all_outputs() -> list[str]:
    return list(dict.fromkeys(f for _, _, outs in analyses(raw_inputs()) for f in outs))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def copy_outputs(name: str, files: list[str]) -> dict:
    dst = WORK / name
    out = {}
    for f in files:
        src = ROOT / f
        (dst / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst / f)
        out[f] = {"bytes": src.stat().st_size, "sha256": sha256(src)}
    return out


def cmd_snapshot(a):
    if (WORK / a.name).exists():
        raise SystemExit(f"{WORK / a.name} exists; pick another name")
    files = copy_outputs(a.name, all_outputs())
    (WORK / a.name / "_index.json").write_text(json.dumps({"kind": "snapshot", "taken": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                                           "files": files}, indent=1))
    print(f"snapshot {a.name}: {len(files)} files")


def cmd_rerun(a):
    if (WORK / a.name).exists():
        raise SystemExit(f"{WORK / a.name} exists; pick another name")
    inp = raw_inputs()
    log = {"kind": "rerun", "layout": inp["layout"], "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "steps": []}
    env = {k: v for k, v in __import__("os").environ.items()
           if not any(s in k.upper() for s in ("API_KEY", "TOKEN", "SECRET"))}
    env["TRIAGE_LLM"] = "off"                   # belt and braces: nothing here may reach a provider
    for name, cmd, outs in analyses(inp):
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        step = {"name": name, "cmd": [str(Path(c).relative_to(ROOT)) if str(c).startswith(str(ROOT)) else c
                                      for c in cmd[1:]], "returncode": r.returncode, "wall_s": round(time.time() - t0, 1)}
        log["steps"].append(step)
        print(json.dumps(step), flush=True)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:], file=sys.stderr)
            raise SystemExit(f"{name} failed")
    files = copy_outputs(a.name, all_outputs())
    log["files"] = files
    (WORK / a.name / "_index.json").write_text(json.dumps(log, indent=1))


# ----------------------------------------------------------------------------- compare

def _text(p: Path) -> str:
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            return fh.read()
    return p.read_text(encoding="utf-8")


def _rewrite_paths(s: str) -> tuple[str, int]:
    n = 0
    for old, new in PATHS.items():
        n += s.count(old)
        s = s.replace(old, new)
    return s, n


def _diff(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((f"{path}.{k}", a.get(k, "<missing>"), b.get(k, "<missing>"), k))
            elif a[k] != b[k]:
                if k in WALL_CLOCK:
                    out.append((f"{path}.{k}", a[k], b[k], k))
                else:
                    _diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                _diff(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append((path, a, b, path.rsplit(".", 1)[-1]))


def compare_file(rel: str, pa: Path, pb: Path) -> dict:
    res = {"file": rel}
    if not pa.exists() or not pb.exists():
        res["status"] = "missing in " + ("both" if not pa.exists() and not pb.exists() else ("A" if not pa.exists() else "B"))
        return res
    if pa.read_bytes() == pb.read_bytes():
        res["status"] = "byte_identical"
        return res
    ta, tb = _text(pa), _text(pb)
    ta, na = _rewrite_paths(ta)
    tb, nb = _rewrite_paths(tb)
    res["old_path_strings"] = {"A": na, "B": nb}
    n_paths = na != nb
    if ta == tb:
        res["status"] = "identical_after_path_rewrite" if n_paths else (
            "identical_after_decompression" if pa.suffix == ".gz" else "identical_text")
        if pa.suffix == ".gz":
            res["note"] = "gzip header holds the write time; the decompressed text is compared"
        return res
    if rel.endswith(".jsonl") or rel.endswith(".jsonl.gz"):
        la, lb = ta.splitlines(), tb.splitlines()
        if len(la) != len(lb):
            res["status"] = "differs"
            res["other"] = [f"line count {len(la)} vs {len(lb)}"]
            return res
        diffs = []
        lines_diff = 0
        for i, (x, y) in enumerate(zip(la, lb)):
            if x != y:
                lines_diff += 1
                _diff(json.loads(x), json.loads(y), f"[line {i + 1}]", diffs)
        res["lines"] = len(la)
        res["lines_differing"] = lines_diff
    else:
        diffs = []
        _diff(json.loads(ta), json.loads(tb), "$", diffs)
    wall, explained, other = {}, {}, []
    for path, x, y, key in diffs:
        if key in WALL_CLOCK:
            wall[key] = wall.get(key, 0) + 1
        elif (rel, key) in EXPLAINED:
            e = explained.setdefault(key, {"count": 0, "values": set(), "why": EXPLAINED[(rel, key)]})
            e["count"] += 1
            e["values"].add(f"{x} -> {y}")
        else:
            other.append({"where": path, "A": x, "B": y})
    res["wall_clock_fields_differing"] = wall
    if explained:
        res["explained"] = {k: dict(v, values=sorted(v["values"])) for k, v in explained.items()}
    res["unexplained"] = other[:20]
    res["unexplained_count"] = len(other)
    res["status"] = ("differs" if other else
                     "equal_except_wall_clock_and_explained" if explained else "equal_except_wall_clock")
    return res


def cmd_compare(a):
    ia = json.loads((WORK / a.a / "_index.json").read_text())
    ib = json.loads((WORK / a.b / "_index.json").read_text())
    files = sorted(set(ia["files"]) | set(ib["files"]))
    rows = [compare_file(f, WORK / a.a / f, WORK / a.b / f) for f in files]
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    out = {"A": a.a, "B": a.b, "A_index": {k: v for k, v in ia.items() if k != "files"},
           "B_index": {k: v for k, v in ib.items() if k != "files"},
           "rules": {"wall_clock_fields_ignored": WALL_CLOCK, "path_rewrites_applied_to_both": PATHS},
           "status_counts": counts, "unexplained_total": sum(r.get("unexplained_count", 0) for r in rows),
           "files": rows}
    text = json.dumps(out, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(json.dumps({"A": a.a, "B": a.b, "status_counts": counts, "unexplained_total": out["unexplained_total"]}))
    for r in rows:
        if r["status"] != "byte_identical":
            print(f"  {r['status']:40s} {r['file']}  wall={r.get('wall_clock_fields_differing')} "
                  f"old_paths={r.get('old_path_strings')} explained={list(r.get('explained', {}))} "
                  f"unexplained={r.get('unexplained_count')}")
            for u in r.get("unexplained", [])[:5]:
                print("      ", json.dumps(u, default=str)[:300])


def cmd_restore(a):
    """Copy WORK/NAME back over the repo (bytes and modification times). With --rewrite-paths, text files get the
    old raw paths replaced by the new ones, as the scripts now write them; nothing else in them changes."""
    idx = json.loads((WORK / a.name / "_index.json").read_text())
    done = {"restored": 0, "rewritten": []}
    for f, v in idx["files"].items():
        src, dst = WORK / a.name / f, ROOT / f
        assert sha256(src) == v["sha256"], f
        shutil.copy2(src, dst)
        done["restored"] += 1
        if a.rewrite_paths and dst.suffix in (".json", ".jsonl", ".md"):
            s, n = _rewrite_paths(dst.read_text(encoding="utf-8"))
            if n:
                st = src.stat()
                dst.write_text(s, encoding="utf-8")
                __import__("os").utime(dst, (st.st_atime, st.st_mtime))
                done["rewritten"].append({"file": f, "replacements": n})
    print(json.dumps(done))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("snapshot", "rerun"):
        sub.add_parser(c).add_argument("name")
    p = sub.add_parser("restore")
    p.add_argument("name")
    p.add_argument("--rewrite-paths", action="store_true")
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--out")
    a = ap.parse_args(argv)
    {"snapshot": cmd_snapshot, "rerun": cmd_rerun, "compare": cmd_compare, "restore": cmd_restore}[a.cmd](a)


if __name__ == "__main__":
    main()
