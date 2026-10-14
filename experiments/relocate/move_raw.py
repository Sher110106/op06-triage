"""Move the raw benchmark outputs that lived under experiments/ into results/raw/, and prove no byte changed.

  python experiments/relocate/move_raw.py inventory NAME   SHA-256 of every file under experiments/ and results/
                                                            (not results/scratch, not __pycache__) -> WORK/inventory_NAME.json
  python experiments/relocate/move_raw.py move             do the moves below; write moves.json next to this file
  python experiments/relocate/move_raw.py rewrite-paths    replace the old raw paths with the new ones inside the run
                                                            records that name them (list in REWRITE); append to moves.json
  python experiments/relocate/move_raw.py verify A B       compare two inventories; every change must be a recorded
                                                            move, a recorded path rewrite, or a file in EDITED

What moves (one folder per run, as the rest of results/raw/):
  experiments/llm_tail/runs/<tag>.jsonl      -> results/raw/llm-tail-<tag>/raw_calls.jsonl   (<tag>.log -> stdout.log)
  experiments/llm_confirm/runs/<tag>.jsonl   -> results/raw/llm-confirm-<tag>/raw_calls.jsonl
  experiments/llm_tail/ledger.jsonl          -> results/raw/llm-ledger/ledger.jsonl   (one ledger shared by all LLM runs)
  experiments/llm_tail/keycheck.jsonl        -> results/raw/llm-ledger/keycheck.jsonl
  experiments/{llm_tail,llm_confirm}/run_summaries.jsonl -> results/raw/llm-ledger/run_summaries.jsonl (joined in time
                                                            order: the llm_tail lines, then the llm_confirm lines)
  experiments/encoder/ett32A/*               -> results/raw/encoder-ett32A/*
Each new run folder gets a meta.json: where each file came from, its SHA-256, and facts read from the files
themselves (model, config, call count, first and last ledger time). Files are renamed, not copied, so their bytes
and modification times stay as they were.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT / "results" / "raw"
WORK = ROOT / "results" / "scratch" / "relocate"
RECORD = HERE / "moves.json"
TODAY = "2026-10-14"

LLM_SRC = {"llm-tail": ROOT / "experiments/llm_tail/runs", "llm-confirm": ROOT / "experiments/llm_confirm/runs"}
LLM_SCRIPT = {"llm-tail": "experiments/llm_tail/run_llm.py", "llm-confirm": "experiments/llm_confirm/run_llm_B.py"}
LLM_PREREG = {"llm-tail": "experiments/llm_tail/PREREGISTRATION.md",
              "llm-confirm": "experiments/llm_confirm/PREREGISTRATION.md"}
LLM_INPUTS = {"llm-tail": ["data/op06/dev.jsonl", "data/op06/dev_noisy.jsonl"],
              "llm-confirm": ["experiments/llm_confirm/rows_B.jsonl"]}
ENC_SRC = ROOT / "experiments/encoder/ett32A"
LEDGER_DIR = RAW / "llm-ledger"
SINGLE = {ROOT / "experiments/llm_tail/ledger.jsonl": LEDGER_DIR / "ledger.jsonl",
          ROOT / "experiments/llm_tail/keycheck.jsonl": LEDGER_DIR / "keycheck.jsonl"}
JOIN_DST = LEDGER_DIR / "run_summaries.jsonl"
JOIN_SRC = [ROOT / "experiments/llm_tail/run_summaries.jsonl", ROOT / "experiments/llm_confirm/run_summaries.jsonl"]

# old raw path -> new raw path, as recorded inside run records
PATHS = {
    "experiments/llm_tail/runs/luna_V3_none.jsonl": "results/raw/llm-tail-luna_V3_none/raw_calls.jsonl",
    "experiments/llm_tail/runs/haiku_V3_low.jsonl": "results/raw/llm-tail-haiku_V3_low/raw_calls.jsonl",
    "experiments/llm_confirm/runs/luna_B.jsonl": "results/raw/llm-confirm-luna_B/raw_calls.jsonl",
    "experiments/llm_confirm/runs/haiku_B.jsonl": "results/raw/llm-confirm-haiku_B/raw_calls.jsonl",
}
# run records and notes that name an old raw path. Pre-registrations are left as written.
REWRITE_GLOBS = ["results/raw/*/meta.json", "results/raw/*/ablation.json", "experiments/ablations/ablations.json",
                 "experiments/chaos/chaos.json", "experiments/chaos/chaos.md"]

# files edited by hand in the same change (scripts that read or write the moved paths, and the manifest builder)
EDITED = [
    "experiments/llm_tail/llm_common.py", "experiments/llm_tail/run_llm.py", "experiments/llm_tail/analyse.py",
    "experiments/llm_confirm/run_llm_B.py", "experiments/llm_confirm/analyse_B.py",
    "experiments/llm_confirm/choose_rule.py", "experiments/encoder/analyse_encoder.py",
    "experiments/encoder/ft_foldA.py", "experiments/routing/analyse_routing.py", "experiments/ablations/common.py",
    "experiments/chaos/run_chaos.py", "experiments/chaos/fake_upstream.py", "results/manifest.json",
]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ----------------------------------------------------------------------------- inventory

def inventory() -> dict:
    out = {}
    for top in ("experiments", "results"):
        for p in sorted((ROOT / top).rglob("*")):
            r = rel(p)
            if not p.is_file() or "__pycache__" in p.parts or r.startswith("results/scratch/") or p.suffix == ".pyc":
                continue
            if p.is_symlink():
                raise SystemExit(f"symlink found: {r}")
            out[r] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    return out


def cmd_inventory(a):
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / f"inventory_{a.name}.json"
    if path.exists():
        raise SystemExit(f"{path} exists")
    inv = inventory()
    path.write_text(json.dumps({"taken": now(), "files": inv}, indent=1))
    print(f"{len(inv)} files -> {rel(path)}")


# ----------------------------------------------------------------------------- move

def plan() -> list[tuple[Path, Path]]:
    moves = []
    for prefix, src in LLM_SRC.items():
        for f in sorted(src.glob("*")):
            if f.suffix == ".jsonl":
                moves.append((f, RAW / f"{prefix}-{f.stem}" / "raw_calls.jsonl"))
            elif f.suffix == ".log":
                moves.append((f, RAW / f"{prefix}-{f.stem}" / "stdout.log"))
            else:
                raise SystemExit(f"unexpected file {rel(f)}")
    for f in sorted(ENC_SRC.glob("*")):
        moves.append((f, RAW / "encoder-ett32A" / f.name))
    moves += list(SINGLE.items())
    return moves


def ledger_times() -> dict[str, tuple[str, str, int]]:
    """tag -> (first ts, last ts, attempts) from the shared ledger."""
    out = {}
    for r in read_jsonl(LEDGER_DIR / "ledger.jsonl"):
        t = r["tag"]
        a, b, n = out.get(t, (r["ts"], r["ts"], 0))
        out[t] = (min(a, r["ts"]), max(b, r["ts"]), n + 1)
    return out


def llm_meta(run_dir: Path, prefix: str, moved: dict, lt: dict) -> dict:
    tag = run_dir.name.removeprefix(prefix + "-")
    recs = read_jsonl(run_dir / "raw_calls.jsonl")
    first, last, attempts = lt.get(tag, (None, None, 0))
    cfg = recs[0].get("cfg") or {}
    return {
        "run_id": run_dir.name, "kind": "llm_calls", "tag": tag,
        "status_note": "every call returned (status ok)" if all(r.get("status") == "ok" for r in recs) else
                       "some calls failed; see status per line",
        "script": LLM_SCRIPT[prefix], "preregistration": LLM_PREREG[prefix],
        "model_requested": recs[0].get("model"), "variant": recs[0].get("variant"), "config": cfg,
        "models_returned": dict(Counter(str(r.get("model_returned")) for r in recs)),
        "providers_returned": dict(Counter(str(r.get("provider_returned")) for r in recs)),
        "rows": len(recs), "status": dict(Counter(str(r.get("status")) for r in recs)),
        "parse_status": dict(Counter(str(r.get("parse_status")) for r in recs)),
        "input_files": LLM_INPUTS[prefix],
        "created": first, "finished": last, "created_source": "first and last ledger line for this tag (local time, IST)",
        "ledger": rel(LEDGER_DIR / "ledger.jsonl"), "ledger_attempts": attempts,
        "files": {name: moved[rel(run_dir / name)] for name in ("raw_calls.jsonl", "stdout.log")
                  if rel(run_dir / name) in moved},
        "meta_written": f"{TODAY} by experiments/relocate/move_raw.py when the files moved here; the run itself "
                        "wrote no meta.json",
    }


def encoder_meta(moved: dict) -> dict:
    d = RAW / "encoder-ett32A"
    log = json.loads((d / "ft_ett32A.json").read_text())
    return {
        "run_id": d.name, "kind": "encoder_finetune", "tag": "ett32A",
        "script": "experiments/encoder/ft_foldA.py (ran on the remote CPU box; outputs copied here)",
        "preregistration": "experiments/encoder/PREREGISTRATION.md", "analysis": "experiments/encoder/analyse_encoder.py",
        "model": log.get("model"), "model_revision": None,
        "model_revision_note": "the Hugging Face revision was not recorded",
        "seed": None, "seed_note": "ft_ett32A.json does not record the seed; ft_foldA.py's default is 13",
        "config": {k: log.get(k) for k in ("threads", "max_len", "bs", "lr", "epochs_planned", "freeze_emb")},
        "train_rows": log.get("train_rows"), "holdout_rows": log.get("holdout_rows"),
        "input_files": ["data/op06/train.jsonl.gz", "data/op06/dev.jsonl", "data/op06/dev_noisy.jsonl",
                        "data/op06/hostile.jsonl"],
        "created": None, "created_note": "trained on 14 Oct 2026 after the pre-registration commit (14:36 IST) and "
                                         "copied here before 14:45 IST; no timestamp inside the files",
        "files": {Path(k).name: v for k, v in moved.items() if k.startswith(rel(d) + "/")},
        "meta_written": f"{TODAY} by experiments/relocate/move_raw.py when the files moved here",
    }


def ledger_meta(moved: dict, joined: dict) -> dict:
    recs = read_jsonl(LEDGER_DIR / "ledger.jsonl")
    by = Counter((r["provider"], r["model"], r["tag"]) for r in recs)
    return {
        "run_id": "llm-ledger", "kind": "llm_ledger",
        "about": "Not a run. One line per API attempt (failed ones too) across every LLM run in this repo, written "
                 "by experiments/llm_tail/llm_common.ledger_write; run_llm.py and run_llm_B.py both append here, so "
                 "spend caps count all runs. keycheck.jsonl: OpenRouter key balance before and after each run. "
                 "run_summaries.jsonl: one summary line per run.",
        "attempts": len(recs), "attempts_by_provider_model_tag": [
            {"provider": p, "model": m, "tag": t, "attempts": n} for (p, m, t), n in sorted(by.items())],
        "status": dict(Counter(str(r.get("status")) for r in recs)),
        "first": min(r["ts"] for r in recs), "last": max(r["ts"] for r in recs),
        "prior_spend_not_in_this_ledger": "the 13 Oct probe's spend (Anthropic $0.4318, OpenRouter $1.2858) is "
                                          "counted as llm_common.PRIOR_SPEND; its ledger lives outside this repo",
        "files": {**{Path(k).name: v for k, v in moved.items() if k.startswith(rel(LEDGER_DIR) + "/")},
                  "run_summaries.jsonl": joined},
        "meta_written": f"{TODAY} by experiments/relocate/move_raw.py when the files moved here",
    }


def cmd_move(_a):
    if RECORD.exists():
        raise SystemExit(f"{rel(RECORD)} exists: the move has been done")
    moves = plan()
    assert moves, "nothing to move"
    for old, new in moves:
        if new.exists():
            raise SystemExit(f"target exists: {rel(new)}")
    for src in JOIN_SRC:
        assert src.exists(), rel(src)
    if JOIN_DST.exists():
        raise SystemExit(f"target exists: {rel(JOIN_DST)}")
    moved = {}
    for old, new in moves:
        before = {"bytes": old.stat().st_size, "sha256": sha256(old), "mtime": old.stat().st_mtime}
        new.parent.mkdir(parents=True, exist_ok=True)
        os.rename(old, new)
        after = {"bytes": new.stat().st_size, "sha256": sha256(new), "mtime": new.stat().st_mtime}
        assert before == after, (rel(old), before, after)
        moved[rel(new)] = {"moved_from": rel(old), "bytes": after["bytes"], "sha256": after["sha256"],
                           "mtime": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(after["mtime"]))}
    # join the two run_summaries files: bytes of the result = bytes of the sources, in order
    parts = [p.read_bytes() for p in JOIN_SRC]
    assert all(b.endswith(b"\n") for b in parts)
    JOIN_DST.write_bytes(b"".join(parts))
    assert JOIN_DST.read_bytes() == b"".join(parts)
    joined = {"joined_from": [{"path": rel(p), "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(),
                               "lines": b.count(b"\n")} for p, b in zip(JOIN_SRC, parts)],
              "bytes": JOIN_DST.stat().st_size, "sha256": sha256(JOIN_DST),
              "check": "bytes equal the sources joined in the order listed"}
    for p in JOIN_SRC:
        p.unlink()
    for d in (*LLM_SRC.values(), ENC_SRC):
        d.rmdir()                                   # fails loudly if anything was left behind
    # meta.json for each new folder
    lt = ledger_times()
    metas = {}
    for prefix in LLM_SRC:
        for d in sorted(RAW.glob(f"{prefix}-*")):
            if (d / "raw_calls.jsonl").exists():
                metas[rel(d / "meta.json")] = llm_meta(d, prefix, moved, lt)
    metas[rel(RAW / "encoder-ett32A" / "meta.json")] = encoder_meta(moved)
    metas[rel(LEDGER_DIR / "meta.json")] = ledger_meta(moved, joined)
    for p, m in metas.items():
        assert not (ROOT / p).exists(), p
        (ROOT / p).write_text(json.dumps(m, indent=1) + "\n")
    RECORD.write_text(json.dumps({
        "done": now(), "script": "experiments/relocate/move_raw.py",
        "rule": "raw benchmark outputs (every run, including failed ones) live under results/raw/",
        "moves": [{"from": v["moved_from"], "to": k, "bytes": v["bytes"], "sha256": v["sha256"],
                   "mtime": v["mtime"]} for k, v in moved.items()],
        "joined": {rel(JOIN_DST): joined},
        "meta_written": sorted(metas),
        "path_rewrites": None}, indent=1) + "\n")
    print(json.dumps({"moved": len(moved), "joined": 1, "meta_written": len(metas)}))


# ----------------------------------------------------------------------------- rewrite paths

def cmd_rewrite_paths(_a):
    rec = json.loads(RECORD.read_text())
    if rec.get("path_rewrites") is not None:
        raise SystemExit("path rewrites already done")
    done = []
    skip = set(rec["meta_written"])                 # written by the move: their "moved_from" must keep the old path
    for g in REWRITE_GLOBS:
        for p in sorted(ROOT.glob(g)):
            if rel(p) in skip:
                continue
            s = p.read_text(encoding="utf-8")
            n = {old: s.count(old) for old in PATHS if old in s}
            if not n:
                continue
            before = sha256(p)
            for old, new in PATHS.items():
                s = s.replace(old, new)
            p.write_text(s, encoding="utf-8")
            if p.suffix == ".json":
                json.loads(s)                       # still valid JSON
            done.append({"file": rel(p), "replacements": n, "sha256_before": before, "sha256_after": sha256(p)})
    rec["path_rewrites"] = {"done": now(), "map": PATHS,
                            "why": "these run records name the raw file a run read; the bytes of that file are "
                                   "unchanged (see moves), only its path is new",
                            "files": done}
    RECORD.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"files_rewritten": len(done), "replacements": sum(sum(d["replacements"].values()) for d in done)}))


# ----------------------------------------------------------------------------- verify

def cmd_verify(a):
    A = json.loads((WORK / f"inventory_{a.a}.json").read_text())["files"]
    B = json.loads((WORK / f"inventory_{a.b}.json").read_text())["files"]
    rec = json.loads(RECORD.read_text())
    moves = {m["from"]: m for m in rec["moves"]}
    joined_src = {j["path"]: (dst, j) for dst, v in rec["joined"].items() for j in v["joined_from"]}
    rewritten = {d["file"]: d for d in (rec.get("path_rewrites") or {}).get("files", [])}
    meta_new = set(rec["meta_written"])
    edited = set(EDITED)
    res = {"unchanged": 0, "moved_same_bytes": 0, "joined_same_bytes": 0, "path_rewritten": [], "edited": [],
           "added": [], "tooling": [], "problems": []}
    for f, v in A.items():
        if f in B:
            if B[f]["sha256"] == v["sha256"]:
                res["unchanged"] += 1
            elif f in rewritten and rewritten[f]["sha256_before"] == v["sha256"] and rewritten[f]["sha256_after"] == B[f]["sha256"]:
                res["path_rewritten"].append(f)
            elif f in edited:
                res["edited"].append(f)
            elif f.startswith("experiments/relocate/"):
                res["tooling"].append(f)              # these scripts, edited after the first inventory
            else:
                res["problems"].append(f"changed without a record: {f}")
        elif f in moves:
            m = moves[f]
            if B.get(m["to"], {}).get("sha256") == v["sha256"] == m["sha256"]:
                res["moved_same_bytes"] += 1
            else:
                res["problems"].append(f"moved file differs: {f} -> {m['to']}")
        elif f in joined_src:
            dst, j = joined_src[f]
            if j["sha256"] == v["sha256"] and B.get(dst, {}).get("sha256") == rec["joined"][dst]["sha256"]:
                res["joined_same_bytes"] += 1
            else:
                res["problems"].append(f"joined file differs: {f}")
        else:
            res["problems"].append(f"missing without a record: {f}")
    targets = {m["to"] for m in rec["moves"]} | set(rec["joined"])
    for f in B:
        if f not in A and f not in targets:
            kind = "new meta.json" if f in meta_new else "new file"
            res["added"].append(f"{f} ({kind})")
    out = {"A": a.a, "B": a.b, "A_files": len(A), "B_files": len(B),
           "summary": {k: (len(v) if isinstance(v, list) else v) for k, v in res.items()}, **res}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["summary"]))
    for k in ("problems", "edited", "added"):
        for x in res[k]:
            print(f"  {k}: {x}")
    if res["problems"]:
        sys.exit(1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inventory").add_argument("name")
    sub.add_parser("move")
    sub.add_parser("rewrite-paths")
    p = sub.add_parser("verify")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--out")
    a = ap.parse_args(argv)
    {"inventory": cmd_inventory, "move": cmd_move, "rewrite-paths": cmd_rewrite_paths, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
