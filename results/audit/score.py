"""Independent OP-06 scorer, written from the problem statement (not harness/eval.py, not grader.py).

usage: python results/audit/score.py PRED.jsonl [--json OUT]
"""
import json, math, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/op06"


def rows(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def valid(p):
    if not isinstance(p, dict):
        return False
    for k in ("id", "intent", "action", "confidence", "needs_human"):
        if k not in p:
            return False
    if not (isinstance(p["id"], str) and isinstance(p["intent"], str) and isinstance(p["action"], str)):
        return False
    if not isinstance(p["needs_human"], bool):
        return False
    c = p["confidence"]
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or not 0 <= c <= 1:
        return False
    for k in ("cost_usd", "latency_ms"):
        if k in p:
            v = p[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                return False
    return True


def acc(gs, ps):
    return sum(g == p for g, p in zip(gs, ps)) / len(gs)


def mf1(gs, ps):
    classes = sorted(set(gs))
    f = []
    for c in classes:
        tp = sum(1 for g, p in zip(gs, ps) if g == c and p == c)
        fp = sum(1 for g, p in zip(gs, ps) if g != c and p == c)
        fn = sum(1 for g, p in zip(gs, ps) if g == c and p != c)
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(f) / len(f)


def score(pred_path):
    clean, noisy, hostile = rows(D / "dev.jsonl"), rows(D / "dev_noisy.jsonl"), rows(D / "hostile.jsonl")
    P = rows(pred_path)
    ids = [p.get("id") for p in P]
    dup = len(ids) - len(set(ids))
    pm = {p["id"]: p for p in P}
    lab = clean + noisy
    out = {"n_pred": len(P), "duplicate_ids": dup,
           "unknown_ids": len(set(ids) - {r["id"] for r in lab + hostile})}
    get = lambda r, k: (pm.get(r["id"]) or {}).get(k)
    for name, part in (("all", lab), ("clean", clean), ("noisy", noisy)):
        gi = [r["intent"] for r in part]; pi = [get(r, "intent") for r in part]
        ga = [r["action"] for r in part]; pa = [get(r, "action") for r in part]
        out[f"intent_acc_{name}"] = acc(gi, pi)
        out[f"macro_f1_{name}"] = mf1(gi, pi)
        out[f"action_acc_{name}"] = acc(ga, pa)
    out["noisy_gap"] = out["action_acc_clean"] - out["action_acc_noisy"]
    # top-70 error ratio, ties at the cut kept; invalid or missing predictions count as conf 0 and wrong
    cr = []
    for r in lab:
        p = pm.get(r["id"])
        if p is not None and valid(p):
            cr.append((float(p["confidence"]), p["action"] == r["action"]))
        else:
            cr.append((0.0, False))
    cr.sort(key=lambda x: -x[0])
    k = max(1, round(0.7 * len(cr)))
    thr = cr[k - 1][0]
    top = [ok for c, ok in cr if c >= thr]
    e_all = 1 - sum(ok for _, ok in cr) / len(cr)
    e_top = 1 - sum(top) / len(top)
    out["top70_kept"] = len(top)
    out["top70_threshold"] = thr
    out["top70_error_ratio"] = e_top / e_all
    every = lab + hostile
    out["schema_valid"] = sum(1 for r in every if r["id"] in pm and valid(pm[r["id"]]))
    out["schema_inputs"] = len(every)
    out["schema_valid_pct"] = 100 * out["schema_valid"] / len(every)
    sc = [r for r in lab if r.get("permitted_actions")]
    out["policy_scored"] = len(sc)
    out["policy_violations"] = sum(1 for r in sc if get(r, "action") is not None and get(r, "action") not in r["permitted_actions"])
    out["policy_violation"] = out["policy_violations"] / len(sc)
    out["gold_policy_violation"] = sum(1 for r in sc if r["action"] not in r["permitted_actions"]) / len(sc)
    out["needs_human_share_labelled"] = sum(1 for r in lab if get(r, "needs_human")) / len(lab)
    hp = [pm.get(r["id"]) for r in hostile]
    out["hostile_needs_human"] = sum(1 for p in hp if p and p.get("needs_human"))
    return out


if __name__ == "__main__":
    res = score(sys.argv[1])
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, indent=1))
