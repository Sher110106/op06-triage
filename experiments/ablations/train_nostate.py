"""Ablation 6: the CPU model retrained without the dialogue-state block in the action model.

A copy of triage.classical.train with one switch. `--state off` drops the state features (last and previous
observed action, actions seen, counts, position and shape) from the action model's input; the text views stay.
Everything else is as shipped: fold A (seed 0) fits the vectorisers, intent, flow and action models; fold B fits
the temperatures, the stacked confidence model and the routing threshold. The stacker keeps its one state feature
(the action-turn count, `state[:, -12]`), so only the action model changes. `--state on` runs the unchanged copy,
as a check that the copy reproduces the shipped model.

  python experiments/ablations/train_nostate.py --state off   # -> experiments/ablations/models/classical_nostate.joblib
  python experiments/ablations/train_nostate.py --state on    # -> experiments/ablations/models/classical_state_copy.joblib
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from triage import classical as C  # noqa: E402
from triage import text as T  # noqa: E402
from triage.data import conv_id, project  # noqa: E402

MODELS = HERE / "models"
OUT = {"off": MODELS / "classical_nostate.joblib", "on": MODELS / "classical_state_copy.joblib"}


def action_matrix_nostate(vecs, rows, views_all):
    """triage.classical._action_matrix without the state block."""
    return sp.hstack([vecs["all_w"].transform(views_all),
                      vecs["lastagent_w"].transform([T.view_last_agent(r) for r in rows]),
                      vecs["lastcust_w"].transform([T.view_last_customer(r) for r in rows]),
                      vecs["last3_w"].transform([T.view_last_k(r) for r in rows])]).tocsr() * np.float32(0.5)


class NoStatePredictor(C.Predictor):
    """Predictor for a bundle whose action model was fit without the state block (bundle["action_state"] False)."""

    def raw_probs(self, rows, st=None):
        if self.b.get("action_state", True):
            return super().raw_probs(rows, st)
        b = self.b
        va = [T.view_all(r) for r in rows]
        xi = C._intent_matrix(b["vecs"], va)
        pi = C.softmax(b["svc"].decision_function(xi), b["T_i"])
        pf = C.softmax(b["fsvc"].decision_function(xi), b["T_f"])
        raw = b["action_model"].predict_proba(action_matrix_nostate(b["vecs"], rows, va))
        pa = np.zeros((len(rows), len(self.acts)))
        for j, c in enumerate(b["action_model"].classes_):
            pa[:, self._aidx[c]] = raw[:, j]
        return pa, pi, pf


def train(out: Path, use_state: bool, log=print) -> dict:
    """triage.classical.train, copied; the only change is `use_state` in the action model's input."""
    import joblib
    from scipy.optimize import minimize_scalar
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.svm import LinearSVC

    t_start = time.time()
    A, B = C.folds()
    XA, XB = [project(r) for r in A], [project(r) for r in B]
    log(f"fold A {len(A)} rows, fold B {len(B)} rows, action state {'on' if use_state else 'off'}")

    def word():
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=150000, dtype=np.float32)

    vecs = {"all_w": word(),
            "all_c": TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                                     max_features=150000, dtype=np.float32),
            "lastagent_w": TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, dtype=np.float32),
            "lastcust_w": TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, dtype=np.float32),
            "last3_w": TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, dtype=np.float32)}
    view_fn = {"all_w": T.view_all, "all_c": T.view_all, "lastagent_w": T.view_last_agent,
               "lastcust_w": T.view_last_customer, "last3_w": T.view_last_k}
    for name, vec in vecs.items():
        vec.fit([view_fn[name](r) for r in XA])
        if hasattr(vec, "stop_words_"):
            del vec.stop_words_
    vA, vB = [T.view_all(r) for r in XA], [T.view_all(r) for r in XB]
    sA = np.vstack([T.state_dense(r) for r in XA]).astype(np.float32)
    sB = np.vstack([T.state_dense(r) for r in XB]).astype(np.float32)
    log(f"features built {time.time() - t_start:.0f}s")

    def fit_temp(scores, classes, y):
        idx = {c: j for j, c in enumerate(classes)}
        yi = np.array([idx.get(v, -1) for v in y])
        ok = yi >= 0
        nll = lambda t: -np.mean(np.log(C.softmax(scores[ok], t)[np.arange(ok.sum()), yi[ok]] + 1e-12))
        return float(minimize_scalar(nll, bounds=(0.01, 5), method="bounded").x)

    XiA, XiB = C._intent_matrix(vecs, vA), C._intent_matrix(vecs, vB)
    svc = LinearSVC(C=0.3).fit(XiA, [r["intent"] for r in A])
    fsvc = LinearSVC(C=0.3).fit(XiA, [r["flow"] for r in A])
    T_i = fit_temp(svc.decision_function(XiB), svc.classes_, [r["intent"] for r in B])
    T_f = fit_temp(fsvc.decision_function(XiB), fsvc.classes_, [r["flow"] for r in B])
    log(f"intent and flow models {time.time() - t_start:.0f}s (T_i={T_i:.3f}, T_f={T_f:.3f})")

    acts = np.array(sorted({r["action"] for r in A}))
    t0 = time.time()
    XaA = C._action_matrix(vecs, XA, vA, sA) if use_state else action_matrix_nostate(vecs, XA, vA)   # the change
    lr = LogisticRegression(C=10.0, max_iter=300).fit(XaA, [r["action"] for r in A])
    log(f"action model fit {time.time() - t0:.0f}s, {XaA.shape[1]} features, n_iter {int(lr.n_iter_.max())}")

    flow_perm = {}
    for r in A:
        flow_perm[r["flow"]] = set(r["permitted_actions"]) | {C.ALIAS[a] for a in r["permitted_actions"] if a in C.ALIAS}
    perm = np.array([[a in flow_perm[f] for a in acts] for f in fsvc.classes_], dtype=np.float32)

    bundle = {"vecs": vecs, "svc": svc, "fsvc": fsvc, "T_i": T_i, "T_f": T_f, "action_model": lr, "acts": acts,
              "perm": perm, "flow_perm": {f: sorted(p) for f, p in flow_perm.items()}, "alias": C.ALIAS,
              "action_state": use_state}

    pm = NoStatePredictor(bundle, dedupe=False)
    pa, pi, pf = pm._probs(XB)
    ok = (acts[pa.argmax(1)] == np.array([r["action"] for r in B])).astype(int)
    feats = C._conf_features(pa, pi, pf, sB, [r.get("turn_index") or 0 for r in XB])
    stk = LogisticRegression(C=1.0, max_iter=1000)
    oof = cross_val_predict(stk, feats, ok, groups=[conv_id(r["id"]) for r in B], cv=GroupKFold(5),
                            method="predict_proba")[:, 1]
    stk.fit(feats, ok)
    thr = float(np.quantile(oof, 0.30))
    bundle.update({"stacker": stk, "route_threshold": thr})
    meta = {"trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seed": C.SEED, "rows_A": len(A), "rows_B": len(B),
            "B_action_acc": round(float(ok.mean()), 4), "route_threshold": round(thr, 4),
            "train_seconds": round(time.time() - t_start, 1), "action_state": use_state,
            "action_features": int(XaA.shape[1]), "action_n_iter": int(lr.n_iter_.max())}
    bundle["meta"] = meta
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out, compress=3)
    meta["bundle_bytes"] = os.path.getsize(out)
    log(json.dumps(meta))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", choices=["off", "on"], default="off")
    a = ap.parse_args()
    meta = train(OUT[a.state], a.state == "on", log=lambda m: print(m, flush=True))
    (OUT[a.state].with_suffix(".meta.json")).write_text(json.dumps(meta, indent=1))
