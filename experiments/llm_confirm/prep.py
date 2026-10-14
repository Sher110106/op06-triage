"""Inputs for the fold-B confirmation test (no API calls).

  cpu_B_predictions.jsonl  CPU model on fold B; confidence is OUT-OF-FOLD (5-fold by conversation), exactly as
                           train() computes it, so the confidence model never scores rows it was fit on
  cpu_B_details.jsonl      CPU intent top 5, p(action), flow
  rows_B.jsonl             projected fold-B inputs (id, turn_index, context only)
  retrieval_B.json         8 most similar fold-A rows per fold-B row (fold B is never in the index)
  ids_luna.json            every fold-B row
  ids_haiku.json           fold-B rows with CPU p(intent) < 0.95 (the only rows the frozen Haiku rule reads)
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from triage import text as T                                   # noqa: E402
from triage.classical import DEFAULT_BUNDLE, Predictor, _conf_features, folds  # noqa: E402
from triage.data import conv_id, project                        # noqa: E402

P_INTENT_GATE = 0.95


def main():
    A, B = folds()
    XB = [project(r) for r in B]
    pm = Predictor(DEFAULT_BUNDLE, dedupe=False)                 # as in train(): the stacker was fit on these features
    pa, pi, pf = pm._probs(XB)
    acts = pm.acts
    pred_a = acts[pa.argmax(1)]
    ok = (pred_a == np.array([r["action"] for r in B])).astype(int)
    st = np.vstack([T.state_dense(r) for r in XB]).astype(np.float32)
    feats = _conf_features(pa, pi, pf, st, [r.get("turn_index") or 0 for r in XB])
    conf = cross_val_predict(LogisticRegression(C=1.0, max_iter=1000), feats, ok, groups=[conv_id(r["id"]) for r in B],
                             cv=GroupKFold(5), method="predict_proba")[:, 1]
    thr = pm.b["route_threshold"]
    assert abs(float(np.quantile(conf, 0.30)) - thr) < 1e-9, "OOF confidences differ from training"
    with open(HERE / "cpu_B_predictions.jsonl", "w") as fp, open(HERE / "cpu_B_details.jsonl", "w") as fd:
        for k, r in enumerate(XB):
            c = float(min(1.0, max(0.0, round(float(conf[k]), 6))))
            fp.write(json.dumps({"id": r["id"], "intent": str(pm.icls[pi[k].argmax()]), "action": str(pred_a[k]),
                                 "confidence": c, "needs_human": bool(c < thr)}) + "\n")
            top = np.argsort(-pi[k])[:5]
            fd.write(json.dumps({"id": r["id"], "intent_top5": [[str(pm.icls[j]), round(float(pi[k][j]), 5)] for j in top],
                                 "p_action": round(float(pa[k].max()), 5), "flow": str(pm.fcls[pf[k].argmax()])}) + "\n")
    with open(HERE / "rows_B.jsonl", "w") as fh:
        for r in XB:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # retrieval over fold A only (same text and vectorisers as the dev test's build_inputs.py)
    def text(r):
        return "\n".join(f"{t['speaker']}: {(t.get('text') or '')[:8000]}" for t in r["context"])
    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200000)
    Tm = normalize(sp.hstack([vw.fit_transform([text(r) for r in A]), vc.fit_transform([text(r) for r in A])]).tocsr())
    Q = normalize(sp.hstack([vw.transform([text(r) for r in XB]), vc.transform([text(r) for r in XB])]).tocsr())
    out = {}
    for s in range(0, len(XB), 500):
        S = (Q[s:s + 500] @ Tm.T).toarray()
        for i, r in enumerate(XB[s:s + 500]):
            picked, convs = [], set()
            for j in np.argsort(-S[i]):
                cid = conv_id(A[j]["id"])
                if cid in convs:
                    continue
                convs.add(cid)
                picked.append((A[j]["id"], round(float(S[i, j]), 4)))
                if len(picked) == 8:
                    break
            out[r["id"]] = picked
    (HERE / "retrieval_B.json").write_text(json.dumps(out))
    (HERE / "ids_luna.json").write_text(json.dumps([r["id"] for r in XB]))
    gated = [r["id"] for k, r in enumerate(XB) if pi[k].max() < P_INTENT_GATE]
    (HERE / "ids_haiku.json").write_text(json.dumps(gated))
    print(json.dumps({"rows_B": len(XB), "cpu_intent_acc": round(float(np.mean([pm.icls[pi[k].argmax()] == B[k]["intent"] for k in range(len(B))])), 4),
                      "cpu_action_acc": round(float(ok.mean()), 4), "routed_share": round(float(np.mean(conf < thr)), 4),
                      "share_below_dev_t20": round(float(np.mean(conf < 0.642841)), 4), "haiku_rows": len(gated)}))


if __name__ == "__main__":
    main()
