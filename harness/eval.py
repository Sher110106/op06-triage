"""Evaluation harness around the official grader.

  python -m harness.eval predict --system classical --run-id dev-classical-v0
  python -m harness.eval grade   --pred results/raw/<run>/predictions.jsonl [--exclude-ids ids.json]
  python -m harness.eval compare --base A.jsonl --new B.jsonl [--exclude-ids ids.json]

`predict` feeds the projected inputs (id, turn_index, context only) of dev, dev_noisy and hostile to a
system and writes grader-format predictions plus timing. `grade` runs data/op06/grader.py unchanged and
adds what it does not report: 95% intervals from a conversation-level bootstrap, results per noise kind,
the policy-violation rate with the membership alias, the needs_human rate and an accuracy-vs-coverage
table. `compare` reports paired differences between two prediction files with bootstrap intervals.
Metric definitions come from the grader module itself so the numbers cannot drift from it.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from triage.data import DATA, ROOT, conv_id, load_split, project

RAW = ROOT / "results" / "raw"
ALIAS = {"membership-privileges": "membership"}
N_BOOT = 2000


def _grader():
    spec = importlib.util.spec_from_file_location("op06_grader", DATA / "grader.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _grader()


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_preds(path) -> dict[str, dict]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line)
            out[str(p["id"])] = p
    return out


# ----------------------------------------------------------------------------- predict

def load_system(name: str, bundle: str | None):
    if name == "classical":
        from triage.classical import DEFAULT_BUNDLE, Predictor
        path = Path(bundle) if bundle else DEFAULT_BUNDLE
        pred = Predictor(path)
        return pred, {"system": "classical", "bundle": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                      "bundle_sha256": sha256(path), "bundle_meta": pred.b.get("meta")}
    if name == "pipeline":
        import os
        from triage.pipeline import Triage
        tri = Triage.from_env(dict(os.environ, TRIAGE_BUNDLE=bundle) if bundle else None)
        h = tri.health()
        return PipelineSystem(tri), {"system": "pipeline", "bundle": h["bundle"], "bundle_sha256": h["bundle_sha256"],
                                     "bundle_meta": tri.cpu.b.get("meta"), "policy": h["policy"], "llm": h["llm"]}
    raise SystemExit(f"unknown system {name}")


class PipelineSystem:
    """predict_many over Triage.handle_many; detail=True adds the trace (no customer text) as "_detail"."""

    def __init__(self, triage):
        self.triage = triage

    def predict_many(self, reqs: list[dict], detail: bool = False) -> list[dict]:
        out = self.triage.handle_many(reqs, [f"eval-{i}" for i in range(len(reqs))])
        return [dict(resp, _detail=trace) if detail else resp for resp, trace in out]


def cmd_predict(a):
    system, info = load_system(a.system, a.bundle)
    out_dir = RAW / a.run_id
    if out_dir.exists() and not a.force:
        raise SystemExit(f"{out_dir} exists; pass --force to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_split("dev") + load_split("noisy") + load_split("hostile")
    reqs = [project(r) for r in rows]
    t0 = time.perf_counter()
    preds = system.predict_many(reqs, detail=True)
    wall = time.perf_counter() - t0
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as fh, \
            open(out_dir / "details.jsonl", "w", encoding="utf-8") as fd:
        for p in preds:
            detail = p.pop("_detail", None)
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
            fd.write(json.dumps({"id": p["id"], **(detail or {})}, ensure_ascii=False) + "\n")
    meta = {"run_id": a.run_id, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **info,
            "inputs": {k: {"file": f"data/op06/{f}", "sha256": sha256(DATA / f)}
                       for k, f in (("dev", "dev.jsonl"), ("noisy", "dev_noisy.jsonl"), ("hostile", "hostile.jsonl"))},
            "rows": len(reqs), "batch_wall_s": round(wall, 2), "python": platform.python_version(),
            "machine": f"{platform.system()} {platform.machine()}",
            "note": "batch prediction for evaluation; per-request latency is measured by the load test, not here"}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"run_id": a.run_id, "rows": len(reqs), "batch_wall_s": round(wall, 2)}))
    if not a.no_grade:
        a.pred, a.out, a.exclude_ids = out_dir / "predictions.jsonl", out_dir / "report.json", None
        cmd_grade(a)


# ----------------------------------------------------------------------------- grade

def labelled():
    dev, noisy = load_split("dev"), load_split("noisy")
    return dev, noisy, load_split("hostile")


class Scored:
    """Per-row arrays for one prediction file over a fixed row list, so bootstrap draws are vectorised.
    Metric definitions match grader.py: accuracy, macro-F1 over gold classes present, top-70% error
    ratio with ties at the cut kept, missing or schema-invalid predictions scored as wrong with confidence 0."""

    def __init__(self, rows, preds):
        self.n = len(rows)
        noisy = np.array([r["_group"] == "noisy" for r in rows])
        self.noisy = noisy
        labels = sorted({r["intent"] for r in rows} | {str(p.get("intent")) for p in preds.values()})
        li = {x: i for i, x in enumerate(labels)}
        self.n_cls = len(labels)
        self.gi = np.array([li[r["intent"]] for r in rows])
        pi, i_ok, a_ok, conf = [], [], [], []
        for r in rows:
            p = preds.get(str(r["id"]))
            valid = p is not None and G.schema_valid(p, None)
            pi.append(li.get(str(p.get("intent")), -1) if p else -1)
            i_ok.append(p is not None and p.get("intent") == r["intent"])
            a_ok.append(p is not None and p.get("action") == r["action"])
            conf.append(float(p["confidence"]) if valid else 0.0)
        self.pi = np.array(pi)
        self.i_ok, self.a_ok = np.array(i_ok), np.array(a_ok)
        self.conf_ok = np.array([a and (preds.get(str(r["id"])) is not None and G.schema_valid(preds[str(r["id"])], None))
                                 for a, r in zip(a_ok, rows)])
        self.order = np.argsort(-np.array(conf), kind="stable")
        self.conf_sorted = np.array(conf)[self.order]

    def metrics(self, w=None):
        w = np.ones(self.n) if w is None else w
        acc = lambda ok, m=None: float((w * ok)[m].sum() / w[m].sum()) if m is None or w[m].sum() > 0 else None
        nc = ~self.noisy
        out = {"intent_acc": acc(self.i_ok), "action_acc": acc(self.a_ok),
               "intent_acc_clean": acc(self.i_ok, nc), "action_acc_clean": acc(self.a_ok, nc)}
        if self.noisy.any() and w[self.noisy].sum() > 0:
            out["intent_acc_noisy"] = acc(self.i_ok, self.noisy)
            out["action_acc_noisy"] = acc(self.a_ok, self.noisy)
            out["noisy_gap"] = out["action_acc_clean"] - out["action_acc_noisy"]
        # macro-F1 over gold classes with weight > 0
        ok = self.i_ok
        tp = np.bincount(self.gi[ok], w[ok], self.n_cls)
        fn = np.bincount(self.gi[~ok], w[~ok], self.n_cls)
        bad = (~ok) & (self.pi >= 0)
        fp = np.bincount(self.pi[bad], w[bad], self.n_cls)
        present = np.bincount(self.gi, w, self.n_cls) > 0
        prec = np.divide(tp, tp + fp, out=np.zeros(self.n_cls), where=(tp + fp) > 0)
        rec = np.divide(tp, tp + fn, out=np.zeros(self.n_cls), where=(tp + fn) > 0)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(self.n_cls), where=(prec + rec) > 0)
        out["macro_f1"] = float(f1[present].mean())
        # top-70% error ratio, weighted; ties at the cut kept
        ws, oks = w[self.order], self.conf_ok[self.order]
        cum = np.cumsum(ws)
        cut = max(1, int(round(0.70 * cum[-1])))
        thr = self.conf_sorted[np.searchsorted(cum, cut)]
        top = self.conf_sorted >= thr
        top_acc = round(float((ws * oks)[top].sum() / ws[top].sum()), 4)       # grader rounds both before the ratio
        overall = round(float((ws * oks).sum() / cum[-1]), 4)
        out["top70_error_ratio"] = (1 - top_acc) / (1 - overall) if overall < 1 else 0.0
        return out


def _metrics(rows, preds):
    return {k: (round(v, 4) if v is not None else None) for k, v in Scored(rows, preds).metrics().items()}


def _weights(rows, n=N_BOOT, seed=7):
    """Conversation-level bootstrap weights: rows of one conversation (and a noisy row with its clean twin) move together."""
    convs = sorted({conv_id(r["id"]) for r in rows})
    ci = {c: i for i, c in enumerate(convs)}
    idx = np.array([ci[conv_id(r["id"])] for r in rows])
    rng = np.random.default_rng(seed)
    for _ in range(n):
        counts = np.bincount(rng.integers(0, len(convs), len(convs)), minlength=len(convs))
        yield counts[idx].astype(float)


def _boot(rows, scored_list):
    return [[s.metrics(w) for s in scored_list] for w in _weights(rows)]


def _ci(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] if len(v) else None


def _rows(exclude=None):
    dev, noisy, hostile = labelled()
    rows = [dict(r, _group="clean") for r in dev] + [dict(r, _group="noisy") for r in noisy]
    if exclude:
        rows = [r for r in rows if r["id"] not in exclude]
    return rows, dev, noisy, hostile


def extras(preds, exclude=None) -> dict:
    rows, *_ = _rows(exclude)
    point = _metrics(rows, preds)
    draws = [d[0] for d in _boot(rows, [Scored(rows, preds)])]
    ci = {k: _ci([d.get(k) for d in draws]) for k in point}
    lab = [r for r in rows]
    viol_alias = np.mean([preds[r["id"]]["action"] not in (set(r["permitted_actions"]) | {ALIAS[x] for x in r["permitted_actions"] if x in ALIAS})
                          for r in lab if r["id"] in preds and r.get("permitted_actions")])
    gold_alias = np.mean([r["action"] not in (set(r["permitted_actions"]) | {ALIAS[x] for x in r["permitted_actions"] if x in ALIAS})
                          for r in lab if r.get("permitted_actions")])
    kinds = defaultdict(list)
    for r in lab:
        if r["_group"] == "noisy":
            kinds[r["noise_kind"]].append(r)
    per_kind = {k: {"n": len(v), "intent_acc": G.accuracy([(r["intent"], preds.get(r["id"], {}).get("intent")) for r in v]),
                    "action_acc": G.accuracy([(r["action"], preds.get(r["id"], {}).get("action")) for r in v])}
                for k, v in sorted(kinds.items())}
    confs = np.array([preds[r["id"]]["confidence"] if r["id"] in preds else 0.0 for r in lab])
    ok = np.array([r["id"] in preds and preds[r["id"]]["action"] == r["action"] for r in lab])
    order = np.argsort(-confs, kind="stable")
    coverage = {f"{int(c * 100)}%": round(float(ok[order][: int(round(c * len(ok)))].mean()), 4)
                for c in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)}
    needs = [bool(preds[r["id"]].get("needs_human")) for r in lab if r["id"] in preds]
    nh = np.array(needs)
    auto = [o for o, n in zip(ok, needs) if not n]
    return {"rows": len(lab), "excluded": len(exclude or ()), "point": point, "ci95_conversation_bootstrap": ci,
            "bootstrap_draws": N_BOOT, "policy_violation_alias": round(float(viol_alias), 4),
            "gold_policy_violation_alias": round(float(gold_alias), 4), "noise_kinds": per_kind,
            "action_acc_at_coverage": coverage, "needs_human_rate": round(float(nh.mean()), 4),
            "action_acc_when_not_routed": round(float(np.mean(auto)), 4) if auto else None}


def cmd_grade(a):
    pred = Path(a.pred).resolve()
    out = Path(a.out).resolve() if a.out else pred.with_name("report.json")
    official = out.with_name(out.stem + "_official.json")
    subprocess.run([sys.executable, "-I", str(DATA / "grader.py"), "--pred", str(pred), "--gold", str(DATA / "dev.jsonl"),
                    "--noisy", str(DATA / "dev_noisy.jsonl"), "--hostile", str(DATA / "hostile.jsonl"),
                    "--schema", str(DATA / "schema.json"), "--report", str(official), "--quiet"], check=True)
    rep = json.loads(official.read_text())
    exclude = set(json.loads(Path(a.exclude_ids).read_text())) if getattr(a, "exclude_ids", None) else None
    preds = read_preds(pred)
    summary = {"pred_file": str(pred.relative_to(ROOT)) if pred.resolve().is_relative_to(ROOT) else str(pred),
               "pred_sha256": sha256(pred), "grader_version": rep["grader_version"],
               "official": {"intent_acc": rep["intent"]["accuracy_all"], "macro_f1": rep["intent"]["macro_f1_all"],
                            "action_acc": rep["action"]["accuracy_all"], "action_acc_clean": rep["action"]["accuracy_clean"],
                            "action_acc_noisy": rep["action"]["accuracy_noisy"], "noisy_gap": rep["action"]["noisy_gap"],
                            "matched_noisy_gap": rep["action"]["matched_noisy_gap"],
                            "intent_acc_noisy": rep["intent"]["accuracy_noisy"],
                            "schema_valid_pct": round(rep["schema"]["valid_rate_all_inputs"] * 100, 2),
                            "top70_error_ratio": rep["confidence"]["top70_error_ratio"],
                            "distinct_confidences": rep["confidence"]["all"]["distinct_confidences"],
                            "policy_violation": rep["policy"]["violation_rate"],
                            "gold_policy_violation": rep["policy"]["gold_violation_rate"]},
               "bars": {k: v["pass"] for k, v in rep["bars"].items()},
               "extras": extras(preds, exclude)}
    if not exclude:   # the vectorised metrics must reproduce the official grader exactly
        pt, of = summary["extras"]["point"], summary["official"]
        pairs = [("intent_acc", "intent_acc"), ("macro_f1", "macro_f1"), ("action_acc", "action_acc"),
                 ("noisy_gap", "noisy_gap"), ("top70_error_ratio", "top70_error_ratio")]
        summary["harness_matches_grader"] = all(abs(pt[a] - of[b]) <= 1e-4 for a, b in pairs)
        if not summary["harness_matches_grader"]:
            print("WARNING: harness metrics differ from grader.py", {a: (pt[a], of[b]) for a, b in pairs}, file=sys.stderr)
    out.write_text(json.dumps(summary, indent=1))
    o, e = summary["official"], summary["extras"]
    print(json.dumps({"intent": o["intent_acc"], "macro_f1": o["macro_f1"], "action": o["action_acc"],
                      "gap": o["noisy_gap"], "top70": o["top70_error_ratio"], "schema": o["schema_valid_pct"],
                      "ci_action": e["ci95_conversation_bootstrap"]["action_acc"], "needs_human": e["needs_human_rate"],
                      "bars_pass": all(v for v in summary["bars"].values() if v is not None)}))


# ----------------------------------------------------------------------------- compare

def paired(base, new, exclude=None, keys=("intent_acc", "macro_f1", "action_acc", "action_acc_clean",
                                          "action_acc_noisy", "intent_acc_noisy", "noisy_gap", "top70_error_ratio"),
           rows=None):
    """rows: labelled dicts with '_group'; defaults to dev + dev_noisy minus `exclude`."""
    if rows is None:
        rows, *_ = _rows(exclude)
    pb, pn = _metrics(rows, base), _metrics(rows, new)
    draws = _boot(rows, [Scored(rows, base), Scored(rows, new)])
    res = {}
    for k in keys:
        if pb.get(k) is None or pn.get(k) is None:
            continue
        diffs = [d[1][k] - d[0][k] for d in draws if d[0].get(k) is not None and d[1].get(k) is not None]
        res[k] = {"base": pb[k], "new": pn[k], "diff": round(pn[k] - pb[k], 4), "ci95": _ci(diffs),
                  "p_new_better": round(float(np.mean([(x < 0) if k in ("noisy_gap", "top70_error_ratio") else (x > 0)
                                                       for x in diffs])), 4)}
    agree_a = np.mean([base[r["id"]]["action"] == new[r["id"]]["action"] for r in rows])
    return {"rows": len(rows), "excluded": len(exclude or ()), "bootstrap_draws": N_BOOT,
            "action_agreement": round(float(agree_a), 4), "metrics": res}


def cmd_compare(a):
    exclude = set(json.loads(Path(a.exclude_ids).read_text())) if a.exclude_ids else None
    res = paired(read_preds(a.base), read_preds(a.new), exclude)
    res = {"base": str(a.base), "new": str(a.new), **res}
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["metrics"], indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--system", default="classical")
    p.add_argument("--bundle")
    p.add_argument("--run-id", required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-grade", action="store_true")
    p.set_defaults(fn=cmd_predict)
    g = sub.add_parser("grade")
    g.add_argument("--pred", required=True)
    g.add_argument("--out")
    g.add_argument("--exclude-ids")
    g.set_defaults(fn=cmd_grade)
    c = sub.add_parser("compare")
    c.add_argument("--base", required=True)
    c.add_argument("--new", required=True)
    c.add_argument("--exclude-ids")
    c.add_argument("--out")
    c.set_defaults(fn=cmd_compare)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
