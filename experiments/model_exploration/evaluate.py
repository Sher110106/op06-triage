"""Matched reference, frozen selection and confirmation for the encoder experiment."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.model_exploration import model_common as M
from triage import classical as C, text as T, gate
from triage.data import conv_id, project
from triage.pipeline import Triage, Settings
from triage.policy import load as load_policy, DEFAULT_POLICY, PolicyStore
from harness.eval import paired, Scored

WEIGHTS = [1., .75, .5, .25, 0.]  # ties favor the cheaper reference
HEADS = ("action", "intent", "flow")
REF = M.MODEL_DIR / "reference.joblib"


def prepare():
    print(json.dumps(M.record_partitions()), flush=True)
    # Both recipes see the same cleaned observed input. Labels remain targets only.
    clean = lambda name: [{**r, **M.model_input(r)} for r in M.partition(name)]
    fit, calibration = clean("fit"), clean("calibration")
    old = C.folds
    C.folds = lambda: (fit, calibration)
    try:
        C.train(REF)
    finally:
        C.folds = old
    bundle = joblib.load(REF)
    bundle["intent_flow"] = {r["intent"]: r["flow"] for r in fit}
    bundle["experiment_partition"] = "65/10/10/15 train-only grouped re-split"
    bundle["meta"].update(seed=None, original_recipe_seed=C.SEED,
                          split_seeds=[20261020, 20261021, 20261022])
    joblib.dump(bundle, REF, compress=3)
    pm = C.Predictor(bundle, dedupe=False)
    assert all(list(getattr(pm, attr)) == M.labels()[head]
               for attr, head in (("acts", "action"), ("icls", "intent"), ("fcls", "flow")))
    for split in ("selection", "calibration"):
        write_raw_reference(pm, split)


def calibrate_reference():
    policy = load_policy(DEFAULT_POLICY)
    rows = M.partition("calibration")
    clf, thresholds = fit_correctness(reference("calibration"), rows, policy)
    joblib.dump(dict(clf=clf, thresholds=thresholds), M.MODEL_DIR / "reference-calibration.joblib")
    for split in ("selection", "calibration"):
        details = {}
        save_preds("reference", split, predict(reference(split), M.partition(split), clf, thresholds, policy, details))
        M.write_json(M.OUT / "reference" / f"details_{split}.json", details)
    print(json.dumps({"reference_thresholds": thresholds}), flush=True)


def write_raw_reference(pm, split):
    rows = M.partition(split)
    x = [M.model_input(r) for r in rows]
    action, intent, flow = pm.raw_probs(x)
    target = M.OUT / "reference"
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / f"probs_{split}.npz", action=action, intent=intent, flow=flow)
    M.write_json(target / f"ids_{split}.json", [r["id"] for r in rows])
    return dict(action=action, intent=intent, flow=flow)


def reference(split):
    target = M.OUT / "reference"
    path = target / f"probs_{split}.npz"
    if not path.exists():
        return write_raw_reference(C.Predictor(REF, dedupe=False), split)
    assert json.loads((target / f"ids_{split}.json").read_text()) == [r["id"] for r in M.partition(split)]
    return dict(np.load(path))


def neural_logits(tag, split):
    target = M.OUT / tag
    ids = json.loads((target / f"ids_{split}.json").read_text())
    assert ids == [r["id"] for r in M.partition(split)], "neural output row order mismatch"
    value = dict(np.load(target / f"logits_{split}.npz"))
    for h in HEADS:
        assert value[h].shape == (len(ids), len(M.labels()[h]))
        assert np.isfinite(value[h]).all()
    config = json.loads((M.MODEL_DIR / tag / "config.json").read_text())
    assert config["labels"] == M.labels(), "neural class order mismatch"
    return value


def temperature(logits, gold, classes):
    idx = {c: i for i, c in enumerate(classes)}
    yi = np.asarray([idx[y] for y in gold])
    def loss(t):
        p = C.softmax(logits, t)
        return -np.log(p[np.arange(len(yi)), yi] + 1e-12).mean()
    return float(minimize_scalar(loss, bounds=(.05, 5.), method="bounded").x)


def mix(ref, neural, action_weight, intent_weight):
    return {h: (intent_weight if h == "intent" else action_weight) * ref[h]
            + (1 - (intent_weight if h == "intent" else action_weight)) * neural[h] for h in HEADS}


def final_action_probs(raw, rows, policy):
    labels = M.labels()
    pa = C.apply_prior(raw["action"].copy(), raw["flow"], policy.prior_matrix(labels["flow"], labels["action"]))
    for i, r in enumerate(rows):
        observed = T.observed_actions(M.model_input(r))
        mask = policy.masked_actions(observed)
        if mask:
            p = pa[i].copy()
            p[[j for j, a in enumerate(labels["action"]) if a in mask]] = 0
            if p.sum() > 0:
                pa[i] = p / p.sum()
    return pa


def correctness_features(raw, rows, policy):
    x = [M.model_input(r) for r in rows]
    pa = final_action_probs(raw, rows, policy)
    return C._conf_features(pa, raw["intent"], raw["flow"],
                            np.vstack([T.state_dense(r) for r in x]), [r["turn_index"] for r in x]), pa


def fit_correctness(raw, rows, policy):
    features, pa = correctness_features(raw, rows, policy)
    classes = np.asarray(M.labels()["action"])
    chosen = classes[pa.argmax(1)]
    ok = (chosen == [r["action"] for r in rows]).astype(int)
    clf = LogisticRegression(C=1., max_iter=1000)
    oof = cross_val_predict(clf, features, ok, cv=GroupKFold(5),
                            groups=[conv_id(r["id"]) for r in rows], method="predict_proba")[:, 1]
    clf.fit(features, ok)
    # Review rules already force some rows to a person. Their cost is constant;
    # a threshold must optimize the remaining rows, using the service's rounding.
    class Certain:
        def predict_proba(self, features):
            return np.tile([0., 1.], (len(features), 1))
    mandatory_preds = predict(raw, rows, Certain(), {"high": 0., "low": 0.}, policy)
    mandatory = np.asarray([mandatory_preds[r["id"]]["needs_human"] for r in rows])
    oof = np.round(oof, 6)
    tiers = np.asarray([policy.risk.get(policy._canon(a), "low") for a in chosen])
    thresholds = {}
    for tier, cost in (("high", 10.), ("low", 3.)):
        use = (tiers == tier) & ~mandatory
        scores, correct = oof[use], ok[use]
        best = (float("inf"), 0.)
        for t in np.unique(np.r_[0., scores, 1.]):
            review = scores < t
            point = float(review.sum() + cost * np.sum(~review & ~correct.astype(bool)))
            if point < best[0]:
                best = point, float(t)
        thresholds[tier] = best[1]
    return clf, thresholds


class CachedPredictor:
    """Replay exported raw probabilities through the real shared service finish path."""
    def __init__(self, raw, rows, clf):
        labels = M.labels()
        self.acts, self.icls, self.fcls = (np.asarray(labels[h]) for h in HEADS)
        self.raw, self.index, self.clf = raw, {r["id"]: i for i, r in enumerate(rows)}, clf

    state = staticmethod(C.Predictor.state)

    def raw_probs(self, rows, st=None):
        idx = [self.index[r["id"]] for r in rows]
        return tuple(self.raw[h][idx].copy() for h in HEADS)

    def confidence(self, pa, pi, pf, st, turn_index):
        return self.clf.predict_proba(C._conf_features(pa, pi, pf, st, turn_index))[:, 1]

    def intent_probs(self, views):
        # Last-customer secondary-intent routing is disabled by the current default.
        # Do not invent another encoder inference path in this cached offline adapter.
        return np.full((len(views), len(self.icls)), 1 / len(self.icls))


def predict(raw, rows, clf, thresholds, policy, trace_sink=None):
    runner = Triage.__new__(Triage)
    runner.cpu = CachedPredictor(raw, rows, clf)
    runner.settings = Settings(route_second_intent=False)
    runner.intent_flow = joblib.load(REF)["intent_flow"]
    runner._aidx = {a: j for j, a in enumerate(runner.cpu.acts)}
    runner._prior_cache, runner.llm = None, None
    runner.store = PolicyStore(DEFAULT_POLICY)
    selected_policy = replace(policy, thresholds=thresholds)
    work, ask = runner._front([project(r) for r in rows], ["" for _ in rows], selected_policy)
    assert not ask
    outputs = runner._back(work)
    assert not any(trace["error"] for _, trace in outputs)
    if trace_sink is not None:
        for p, trace in outputs:
            trace_sink[p["id"]] = {"gate_confidence": (trace.get("cpu") or {}).get("confidence", 0.),
                                  "gate_flags": trace["flags"], "abstained": bool(trace["abstain_reason"])}
    return {p["id"]: p for p, _ in outputs}


def save_preds(name, split, preds):
    path = M.OUT / f"{name}_{split}.jsonl"
    path.write_text("".join(json.dumps(preds[r["id"]]) + "\n" for r in M.partition(split)))


def select(tag):
    policy = load_policy(DEFAULT_POLICY)
    lab = M.labels()
    cal, selection = M.partition("calibration"), M.partition("selection")
    zcal, zsel = neural_logits(tag, "calibration"), neural_logits(tag, "selection")
    temps = {h: temperature(zcal[h], [r[h] for r in cal], lab[h]) for h in HEADS}
    neural = {split: {h: C.softmax(z[h], temps[h]) for h in HEADS}
              for split, z in (("calibration", zcal), ("selection", zsel))}
    ref = {s: reference(s) for s in ("calibration", "selection")}
    action_scores, intent_scores = {}, {}
    for w in WEIGHTS:
        raw = mix(ref["selection"], neural["selection"], w, w)
        action_scores[w] = float((np.asarray(lab["action"])[final_action_probs(raw, selection, policy).argmax(1)]
                                 == [r["action"] for r in selection]).mean())
        intent_scores[w] = float((np.asarray(lab["intent"])[raw["intent"].argmax(1)]
                                 == [r["intent"] for r in selection]).mean())
    aw, iw = max(WEIGHTS, key=action_scores.get), max(WEIGHTS, key=intent_scores.get)
    settings = {"tag": tag, "temperatures": temps, "classical_action_flow_weight": aw,
                "classical_intent_weight": iw, "selection_action_by_weight": action_scores,
                "selection_intent_by_weight": intent_scores, "policy_version": policy.version,
                "primary_contrast": "selection-frozen ensemble vs matched retrained classical",
                "calibration_note": "Correctness OOF excludes groups only in the final regression; upstream temperatures are shared."}
    fitted = {}
    for name, raw in (("reference", ref["calibration"]), ("encoder", neural["calibration"]),
                       ("ensemble", mix(ref["calibration"], neural["calibration"], aw, iw))):
        clf, cuts = fit_correctness(raw, cal, policy)
        fitted[name] = dict(clf=clf, thresholds=cuts)
        settings.setdefault("thresholds", {})[name] = cuts
        save_preds(name, "calibration", predict(raw, cal, clf, cuts, policy))
        raw_sel = ref["selection"] if name == "reference" else neural["selection"] if name == "encoder" else mix(ref["selection"], neural["selection"], aw, iw)
        save_preds(name, "selection", predict(raw_sel, selection, clf, cuts, policy))
    joblib.dump(fitted, M.MODEL_DIR / f"{tag}-calibration.joblib", compress=3)
    M.write_json(M.OUT / tag / "frozen.json", settings)
    print(json.dumps(settings), flush=True)


def score_rows(rows, preds, policy):
    labelled = [{**r, "_group": "noisy" if str(r["id"]).endswith("-noisy") else "clean"}
                for r in rows if all(isinstance(r.get(k), str) for k in ("intent", "action", "flow"))]
    if not labelled:
        return {"rows": len(rows)}
    metric = Scored(labelled, preds).metrics()
    automatic = [r for r in labelled if not preds[r["id"]]["needs_human"]]
    forbidden = [r for r in automatic if not policy._permitted(preds[r["id"]]["action"], r["flow"])]
    wrong = [r for r in automatic if preds[r["id"]]["action"] != r["action"]]
    cost = len(labelled) - len(automatic) + sum(
        10 if policy.risk.get(policy._canon(preds[r["id"]]["action"]), "low") == "high" else 3 for r in wrong)
    metric.update(rows=len(labelled), reviews=len(labelled) - len(automatic), automatic=len(automatic),
                  wrong_automatic=len(wrong), action_cost_per100=100 * cost / len(labelled),
                  forbidden_automatic=len(forbidden), forbidden_high=sum(policy.risk.get(policy._canon(preds[r["id"]]["action"]), "low") == "high" for r in forbidden))
    return metric


def exact_paired(rows, baseline, candidate, policy, draws=5000):
    """Row-weighted paired differences; resample whole conversations, no display rounding."""
    def events(preds):
        out = []
        for r in rows:
            p = preds[r["id"]]
            review = bool(p["needs_human"])
            high = policy.risk.get(policy._canon(p["action"]), "low") == "high"
            wrong = p["action"] != r["action"]
            forbidden = not review and not policy._permitted(p["action"], r["flow"])
            cost = float(review) + (10 if high else 3) * float(not review and wrong)
            out.append([p["intent"] == r["intent"], not wrong, review, forbidden, forbidden and high, cost * 100])
        return np.asarray(out, dtype=float)
    b, n = events(baseline), events(candidate)
    groups = [conv_id(r["id"]) for r in rows]
    unique = sorted(set(groups)); index = {g: i for i, g in enumerate(unique)}
    sums = np.zeros((len(unique), 13))
    np.add.at(sums, [index[g] for g in groups], np.column_stack([np.ones(len(rows)), b, n]))
    rng = np.random.default_rng(20261024)
    deltas = []
    for start in range(0, draws, 250):
        weights = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=min(250, draws - start))
        total = weights @ sums
        deltas.append((total[:, 7:] - total[:, 1:7]) / total[:, :1])
    boot = np.vstack(deltas)
    names = ("intent_accuracy", "action_accuracy", "review_rate", "forbidden_automatic_rate",
             "forbidden_high_rate", "action_cost_per100")
    return {name: {"base": float(b[:, i].mean()), "new": float(n[:, i].mean()),
                   "diff": float((n[:, i] - b[:, i]).mean()),
                   "ci95": np.quantile(boot[:, i], [.025, .975]).tolist()}
            for i, name in enumerate(names)}


def accuracy_rule_pass(delta):
    """Apply the frozen rule to full-precision values, before presentation rounding."""
    return (delta["action_accuracy"]["diff"] >= .01
            and delta["action_accuracy"]["ci95"][0] > 0
            and delta["intent_accuracy"]["diff"] >= -.01)


def confirm(tag, splits):
    settings = json.loads((M.OUT / tag / "frozen.json").read_text())
    fitted = joblib.load(M.MODEL_DIR / f"{tag}-calibration.joblib")
    policy = load_policy(DEFAULT_POLICY)
    assert policy.version == settings["policy_version"]
    result = {}
    for split in splits:
        rows = M.partition(split)
        zr = neural_logits(tag, split)
        neural = {h: C.softmax(zr[h], settings["temperatures"][h]) for h in HEADS}
        ref = reference(split)
        systems = {"reference": ref, "encoder": neural,
                   "ensemble": mix(ref, neural, settings["classical_action_flow_weight"], settings["classical_intent_weight"])}
        preds = {}
        for name, raw in systems.items():
            f = fitted[name]
            details = {} if name == "reference" else None
            preds[name] = predict(raw, rows, f["clf"], f["thresholds"], policy, details)
            save_preds(name, split, preds[name])
            if details is not None:
                M.write_json(M.OUT / "reference" / f"details_{split}.json", details)
        result[split] = {name: score_rows(rows, p, policy) for name, p in preds.items()}
        if split == "confirmation":
            labelled = [{**r, "_group": "clean"} for r in rows]
            result[split]["primary_paired"] = paired(preds["reference"], preds["ensemble"], rows=labelled,
                                                      keys=("intent_acc", "macro_f1", "action_acc", "top70_error_ratio"))
            delta = exact_paired(rows, preds["reference"], preds["ensemble"], policy)
            result[split]["primary_exact_paired"] = delta
            result[split]["accuracy_rule_pass"] = accuracy_rule_pass(delta)
            result[split]["operational_rule_pass"] = (
                delta["action_cost_per100"]["diff"] <= 2.
                and delta["review_rate"]["diff"] <= .05
                and delta["forbidden_high_rate"]["diff"] <= 0.)
            result[split]["integration_rule_pass"] = (
                result[split]["accuracy_rule_pass"] and result[split]["operational_rule_pass"])
    M.write_json(M.OUT / tag / ("confirmation.json" if "confirmation" in splits else "diagnostics.json"), result)
    print(json.dumps(result, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("prepare")
    sub.add_parser("calibrate-reference")
    sel = sub.add_parser("select"); sel.add_argument("--tag", required=True)
    cf = sub.add_parser("confirm"); cf.add_argument("--tag", required=True)
    cf.add_argument("--splits", nargs="+", default=["confirmation"])
    a = ap.parse_args()
    if a.stage == "prepare": prepare()
    elif a.stage == "calibrate-reference": calibrate_reference()
    elif a.stage == "select": select(a.tag)
    else: confirm(a.tag, a.splits)


if __name__ == "__main__":
    main()
