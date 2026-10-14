"""Choose the Haiku + CPU intent combination rule on dev (the fold-B test then checks it).
Each candidate's parameters are fit on 4/5 of dev conversations and scored on the other 1/5 (5 folds),
so the reported accuracy is out-of-fold. The winner is refit on all of dev and frozen in rule.json."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triage.data import load_split, conv_id
from harness.eval import read_preds

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
rows = load_split("dev") + load_split("noisy")
cpu = read_preds(ROOT / "results/raw/dev-classical-v0/predictions.jsonl")
det = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "results/raw/dev-classical-v0/details.jsonl")}
hk = {}
for l in open(ROOT / "experiments/llm_tail/runs/haiku_V3_low.jsonl"):
    r = json.loads(l); p = r.get("pred")
    if r["status"] == "ok" and p and p["intent"] != "unknown" and p["action"] != "none":
        hk[r["id"]] = p["intent"]
lu = {}
for l in open(ROOT / "experiments/llm_tail/runs/luna_V3_none.jsonl"):
    r = json.loads(l); p = r.get("pred")
    if r["status"] == "ok" and p and p["intent"] != "unknown" and p["action"] != "none":
        lu[r["id"]] = p["intent"]
SRC = {"haiku": hk, "luna": lu}
# intent -> flow from train
flow_of = {r["intent"]: r["flow"] for r in load_split("train")}

def apply(rule, prm, r):
    c = cpu[r["id"]]["intent"]; d = det[r["id"]]
    if rule.startswith("gated_topk_"):            # call the LLM only when CPU p(intent) < t; accept if in CPU top k
        if d["intent_top5"][0][1] >= prm[1]: return c
        h = SRC[rule.removeprefix("gated_topk_")].get(r["id"])
        return h if h is not None and h in [x for x, _ in d["intent_top5"][:prm[0]]] else c
    h = hk.get(r["id"])
    if h is None: return c
    if rule == "cpu": return c
    if rule == "haiku": return h
    if rule == "flow_switch_cpuflow": return h if d["flow"] in prm else c
    if rule == "flow_switch_haikuflow": return h if flow_of.get(h) in prm else c
    if rule == "p_intent_below": return h if d["intent_top5"][0][1] < prm else c
    if rule == "haiku_if_in_cpu_topk": return h if h in [x for x, _ in d["intent_top5"][:prm]] else c
    raise ValueError(rule)

def acc(rule, prm, rs): return float(np.mean([apply(rule, prm, r) == r["intent"] for r in rs]))

def fit(rule, rs):
    if rule in ("cpu", "haiku"): return None
    if rule.startswith("flow_switch"):
        key = (lambda r: det[r["id"]]["flow"]) if rule.endswith("cpuflow") else (lambda r: flow_of.get(hk.get(r["id"])))
        gain = defaultdict(int)
        for r in rs:
            h = hk.get(r["id"]); c = cpu[r["id"]]["intent"]
            if h is None: continue
            gain[key(r)] += (h == r["intent"]) - (c == r["intent"])
        return sorted(f for f, g in gain.items() if g > 0 and f is not None)
    if rule == "p_intent_below":
        grid = np.round(np.arange(0.1, 1.0, 0.05), 2)
        return float(max(grid, key=lambda t: acc(rule, t, rs)))
    if rule == "haiku_if_in_cpu_topk":
        return int(max([2, 3, 4, 5], key=lambda k: acc(rule, k, rs)))
    if rule.startswith("gated_topk_"):
        grid = [(k, t) for k in (2, 3, 4, 5) for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01)]
        return max(grid, key=lambda g: acc(rule, g, rs))

convs = sorted({conv_id(r["id"]) for r in rows}); rng = np.random.default_rng(0); rng.shuffle(convs)
fold = {c: i % 5 for i, c in enumerate(convs)}
rules = ["cpu", "haiku", "flow_switch_cpuflow", "flow_switch_haikuflow", "p_intent_below", "haiku_if_in_cpu_topk",
         "gated_topk_haiku", "gated_topk_luna"]
res = {}
for rule in rules:
    oof = []
    for k in range(5):
        tr = [r for r in rows if fold[conv_id(r["id"])] != k]; te = [r for r in rows if fold[conv_id(r["id"])] == k]
        prm = fit(rule, tr)
        oof += [apply(rule, prm, r) == r["intent"] for r in te]
    full = fit(rule, rows)
    share = float(np.mean([apply(rule, full, r) != cpu[r["id"]]["intent"] for r in rows]))
    if rule == "p_intent_below":
        needs = float(np.mean([det[r["id"]]["intent_top5"][0][1] < full for r in rows]))
    elif rule.startswith("gated_topk_"):
        needs = float(np.mean([det[r["id"]]["intent_top5"][0][1] < full[1] for r in rows]))
    else:
        needs = 0.0 if rule == "cpu" else 1.0
    res[rule] = {"oof_intent_acc": round(float(np.mean(oof)), 4), "params_all_dev": full,
                 "answers_changed": round(share, 4), "share_needing_haiku_call": round(needs, 4)}
    print(rule, res[rule])
# sanity: how often do the two LLMs give the same intent, and how often does each sit in the CPU top 3?
both = [r for r in rows if r["id"] in hk and r["id"] in lu]
res["_llm_intent_agreement"] = round(float(np.mean([hk[r["id"]] == lu[r["id"]] for r in both])), 4)
for name, src in SRC.items():
    res[f"_{name}_in_cpu_top3"] = round(float(np.mean([src[r["id"]] in [x for x, _ in det[r["id"]]["intent_top5"][:3]]
                                                       for r in rows if r["id"] in src])), 4)
print({k: v for k, v in res.items() if k.startswith("_")})
json.dump(res, open(HERE / "rule_selection_dev.json", "w"), indent=1)
