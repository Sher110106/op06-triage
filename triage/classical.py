"""The classical CPU model (v0): the final configuration of the 13 Oct classical probe, made reproducible.

  intent  LinearSVC (C=0.3) on word + char TF-IDF of all 8 turns; probabilities = softmax(decision / T)
  flow    the same model over the 10 flows (feeds the policy prior)
  action  logistic regression (C=10) on four text views + dialogue-state features
  policy  soft prior: p(action) x E_flow[action permitted], with the membership-privileges -> membership alias
  conf    stacked logistic regression that predicts "is the chosen action right?" (fit on a train hold-out)
  route   needs_human = confidence below the 30% quantile of hold-out confidences

Everything is fit on train only: fold A (80% of train conversations) for the models, fold B (20%) for
temperatures, the confidence model and the routing threshold. Dev is never used for fitting.

  python -m triage.classical train [--out models/classical_v0.joblib]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import text as T
from .data import ROOT, conv_id, load_split, project

DEFAULT_BUNDLE = ROOT / "models" / "classical_v0.joblib"
ALIAS = {"membership-privileges": "membership"}
SEED = 0


def softmax(z, temp=1.0):
    z = np.asarray(z, dtype=np.float64) / temp
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def apply_prior(pa, pf, perm):
    """Soft policy prior: p(action) x E_flow[action permitted], renormalised; a row with no mass left stays 0."""
    pa = pa * (pf @ perm)
    s = pa.sum(1, keepdims=True)
    s[s == 0] = 1
    return pa / s


def _intent_matrix(vecs, views_all):
    return sp.hstack([vecs["all_w"].transform(views_all), vecs["all_c"].transform(views_all)]).tocsr() * np.float32(0.7071)


def _action_matrix(vecs, rows, views_all, state):
    tb = sp.hstack([vecs["all_w"].transform(views_all),
                    vecs["lastagent_w"].transform([T.view_last_agent(r) for r in rows]),
                    vecs["lastcust_w"].transform([T.view_last_customer(r) for r in rows]),
                    vecs["last3_w"].transform([T.view_last_k(r) for r in rows])]).tocsr() * np.float32(0.5)
    return sp.hstack([tb, sp.csr_matrix(state)]).tocsr()


def _conf_features(pa, pi, pf, state, turn_index):
    sa, si = np.sort(pa, 1), np.sort(pi, 1)
    ent = -(pa * np.log(pa + 1e-12)).sum(1)
    return np.column_stack([sa[:, -1], sa[:, -1] - sa[:, -2], si[:, -1], si[:, -1] - si[:, -2], pf.max(1), ent,
                            state[:, -12], np.minimum(np.asarray(turn_index), 60) / 60.0])


# ----------------------------------------------------------------------------- training

def folds():
    """Train split by conversation: fold A (80%) fits the models, fold B (20%) tunes them. Seed 0."""
    from sklearn.model_selection import GroupShuffleSplit
    train_rows = load_split("train")
    groups = [conv_id(r["id"]) for r in train_rows]
    a_idx, b_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(train_rows, groups=groups))
    return [train_rows[i] for i in a_idx], [train_rows[i] for i in b_idx]


def train(out: Path = DEFAULT_BUNDLE, log=print) -> dict:
    import joblib
    from scipy.optimize import minimize_scalar
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.svm import LinearSVC

    t_start = time.time()
    A, B = folds()
    XA, XB = [project(r) for r in A], [project(r) for r in B]          # the model only sees projected inputs
    log(f"fold A {len(A)} rows, fold B {len(B)} rows")

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
            del vec.stop_words_                                            # introspection only; keeps the bundle small
    vA, vB = [T.view_all(r) for r in XA], [T.view_all(r) for r in XB]
    sA = np.vstack([T.state_dense(r) for r in XA]).astype(np.float32)
    sB = np.vstack([T.state_dense(r) for r in XB]).astype(np.float32)
    log(f"features built {time.time() - t_start:.0f}s")

    def fit_temp(scores, classes, y):
        idx = {c: j for j, c in enumerate(classes)}
        yi = np.array([idx.get(v, -1) for v in y])
        ok = yi >= 0
        nll = lambda t: -np.mean(np.log(softmax(scores[ok], t)[np.arange(ok.sum()), yi[ok]] + 1e-12))
        return float(minimize_scalar(nll, bounds=(0.01, 5), method="bounded").x)

    XiA, XiB = _intent_matrix(vecs, vA), _intent_matrix(vecs, vB)
    svc = LinearSVC(C=0.3).fit(XiA, [r["intent"] for r in A])
    fsvc = LinearSVC(C=0.3).fit(XiA, [r["flow"] for r in A])
    T_i = fit_temp(svc.decision_function(XiB), svc.classes_, [r["intent"] for r in B])
    T_f = fit_temp(fsvc.decision_function(XiB), fsvc.classes_, [r["flow"] for r in B])
    log(f"intent and flow models {time.time() - t_start:.0f}s (T_i={T_i:.3f}, T_f={T_f:.3f})")

    acts = np.array(sorted({r["action"] for r in A}))
    t0 = time.time()
    lr = LogisticRegression(C=10.0, max_iter=300).fit(_action_matrix(vecs, XA, vA, sA), [r["action"] for r in A])
    log(f"action model fit {time.time() - t0:.0f}s")

    # policy table: flow -> permitted actions, read from train rows (equals guidelines.json resolved), plus alias
    flow_perm = {}
    for r in A:
        flow_perm[r["flow"]] = set(r["permitted_actions"]) | {ALIAS[a] for a in r["permitted_actions"] if a in ALIAS}
    perm = np.array([[a in flow_perm[f] for a in acts] for f in fsvc.classes_], dtype=np.float32)

    bundle = {"vecs": vecs, "svc": svc, "fsvc": fsvc, "T_i": T_i, "T_f": T_f, "action_model": lr, "acts": acts,
              "perm": perm, "flow_perm": {f: sorted(p) for f, p in flow_perm.items()}, "alias": ALIAS}

    # confidence model and routing threshold, on fold B only (no dedupe: training views are raw, as in the probe)
    pm = Predictor(bundle, dedupe=False)
    pa, pi, pf = pm._probs(XB)
    ok = (acts[pa.argmax(1)] == np.array([r["action"] for r in B])).astype(int)
    feats = _conf_features(pa, pi, pf, sB, [r.get("turn_index") or 0 for r in XB])
    stk = LogisticRegression(C=1.0, max_iter=1000)
    oof = cross_val_predict(stk, feats, ok, groups=[conv_id(r["id"]) for r in B], cv=GroupKFold(5),
                            method="predict_proba")[:, 1]
    stk.fit(feats, ok)
    thr = float(np.quantile(oof, 0.30))
    bundle.update({"stacker": stk, "route_threshold": thr})
    meta = {"trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seed": SEED, "rows_A": len(A), "rows_B": len(B),
            "B_action_acc": round(float(ok.mean()), 4), "route_threshold": round(thr, 4),
            "train_seconds": round(time.time() - t_start, 1)}
    bundle["meta"] = meta
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out, compress=3)
    meta["bundle_bytes"] = os.path.getsize(out)
    log(json.dumps(meta))
    return meta


# ----------------------------------------------------------------------------- inference

class Predictor:
    """Loads a bundle once; `predict_many` and `predict_one` return grader-shaped dicts."""

    def __init__(self, bundle, dedupe: bool = True):
        if isinstance(bundle, (str, Path)):
            import joblib
            bundle = joblib.load(bundle)
        self.b = bundle
        self.dedupe = dedupe
        self.icls, self.fcls, self.acts = bundle["svc"].classes_, bundle["fsvc"].classes_, bundle["acts"]
        self._aidx = {a: j for j, a in enumerate(self.acts)}

    def _sanitise(self, req) -> dict:
        ctx = req.get("context") if isinstance(req, dict) else None
        clean = []
        for t in (ctx if isinstance(ctx, list) else [])[-8:]:
            if not isinstance(t, dict):
                continue
            txt = t.get("text") if isinstance(t.get("text"), str) else ""
            if self.dedupe:
                txt = T.dedupe_text(txt)
            spk = t.get("speaker") if t.get("speaker") in ("customer", "agent", "action") else "customer"
            clean.append({"speaker": spk, "text": txt})
        ti = req.get("turn_index") if isinstance(req, dict) else None
        ti = ti if isinstance(ti, int) and not isinstance(ti, bool) and 0 <= ti < 10000 else len(clean)
        return {"id": str(req.get("id", "")) if isinstance(req, dict) else "", "context": clean, "turn_index": ti}

    @staticmethod
    def state(rows) -> np.ndarray:
        return np.vstack([T.state_dense(r) for r in rows]).astype(np.float32)

    def raw_probs(self, rows, st=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(pa, pi, pf): action probabilities before any policy prior or mask, intent and flow probabilities.
        `st` is self.state(rows) when the caller already has it."""
        b = self.b
        va = [T.view_all(r) for r in rows]
        xi = _intent_matrix(b["vecs"], va)
        pi = softmax(b["svc"].decision_function(xi), b["T_i"])
        pf = softmax(b["fsvc"].decision_function(xi), b["T_f"])
        st = self.state(rows) if st is None else st
        raw = b["action_model"].predict_proba(_action_matrix(b["vecs"], rows, va, st))
        pa = np.zeros((len(rows), len(self.acts)))
        for j, c in enumerate(b["action_model"].classes_):
            pa[:, self._aidx[c]] = raw[:, j]
        return pa, pi, pf

    def intent_probs(self, views: list[str]) -> np.ndarray:
        """The intent model on any text written like T.view_all, for example one turn on its own."""
        return softmax(self.b["svc"].decision_function(_intent_matrix(self.b["vecs"], views)), self.b["T_i"])

    def confidence(self, pa, pi, pf, st, turn_index) -> np.ndarray:
        """The stacker's p(chosen action is right), unrounded."""
        return self.b["stacker"].predict_proba(_conf_features(pa, pi, pf, st, turn_index))[:, 1]

    def _probs(self, rows):
        pa, pi, pf = self.raw_probs(rows)
        return apply_prior(pa, pf, self.b["perm"]), pi, pf

    @staticmethod
    def abstain(rid: str, msg: str) -> dict:
        return {"id": rid, "intent": "unknown", "action": "none", "confidence": 0.0, "needs_human": True,
                "error": {"code": "invalid_input", "message": msg}}

    def predict_many(self, reqs: list[dict], detail: bool = False) -> list[dict]:
        rows = [self._sanitise(q) for q in reqs]
        out: list[dict | None] = [None] * len(rows)
        live = []
        for i, r in enumerate(rows):
            if not any(re.search(r"[a-z0-9]", T.clean_text(t["text"])) for t in r["context"]):
                out[i] = self.abstain(r["id"], "no readable text in context")
            else:
                live.append(i)
        if live:
            lrows = [rows[i] for i in live]
            pa, pi, pf = self._probs(lrows)
            conf = self.confidence(pa, pi, pf, self.state(lrows), [r["turn_index"] for r in lrows])
            for k, i in enumerate(live):
                c = float(min(1.0, max(0.0, round(float(conf[k]), 6))))
                rec = {"id": rows[i]["id"], "intent": str(self.icls[pi[k].argmax()]),
                       "action": str(self.acts[pa[k].argmax()]), "confidence": c,
                       "needs_human": bool(c < self.b["route_threshold"])}
                if detail:
                    top = np.argsort(-pi[k])[:5]
                    rec["_detail"] = {"intent_top5": [[str(self.icls[j]), round(float(pi[k][j]), 5)] for j in top],
                                      "p_action": round(float(pa[k].max()), 5), "p_flow": round(float(pf[k].max()), 5),
                                      "flow": str(self.fcls[pf[k].argmax()])}
                out[i] = rec
        return out

    def predict_one(self, req: dict) -> dict:
        rid = str(req.get("id", "")) if isinstance(req, dict) else ""
        try:
            return self.predict_many([req])[0]
        except Exception as e:                                             # one bad request never takes the service down
            return self.abstain(rid, f"internal: {type(e).__name__}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--out", type=Path, default=DEFAULT_BUNDLE)
    a = ap.parse_args()
    train(a.out, log=lambda m: print(m, flush=True))
