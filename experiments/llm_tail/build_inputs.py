"""Build the inputs of the full-dev LLM test (no API calls):
  rows_all.json            the 2,300 labelled dev + dev_noisy ids, file order
  seen_in_probe_ids.json   the 360 labelled ids the 13 Oct probe already used (excluded from the primary analysis)
  retrieval.json           for every row, the 8 most similar TRAIN rows (word + char TF-IDF, one per conversation)
The retrieval code is the probe's build_retrieval.py applied to all rows instead of the 369-row sample."""
import json
import sys
import time

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import llm_common as C


def text(r):
    return "\n".join(f"{t['speaker']}: {(t.get('text') or '')[:C.MAX_TURN_CHARS]}" for t in r["context"])


def main(probe_sample):
    t0 = time.time()
    d = C.data()
    rows = [C.strip_gold(r) for r in d["dev"] + d["noisy"]]
    (C.WORK / "rows_all.json").write_text(json.dumps([r["id"] for r in rows]))
    sample = json.loads(open(probe_sample).read())
    seen = [x["id"] for x in sample if x["group"] != "hostile"]
    (C.WORK / "seen_in_probe_ids.json").write_text(json.dumps(sorted(seen)))
    tr = d["train"]
    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200000)
    T = normalize(sp.hstack([vw.fit_transform([text(r) for r in tr]), vc.fit_transform([text(r) for r in tr])]).tocsr())
    Q = normalize(sp.hstack([vw.transform([text(r) for r in rows]), vc.transform([text(r) for r in rows])]).tocsr())
    out = {}
    for s in range(0, len(rows), 500):
        S = (Q[s:s + 500] @ T.T).toarray()
        for i, r in enumerate(rows[s:s + 500]):
            picked, convs = [], set()
            for j in np.argsort(-S[i]):
                cid = tr[j]["id"].rsplit("-", 1)[0]
                if cid in convs:
                    continue
                convs.add(cid)
                picked.append((tr[j]["id"], round(float(S[i, j]), 4)))
                if len(picked) == 8:
                    break
            out[r["id"]] = picked
    (C.WORK / "retrieval.json").write_text(json.dumps(out))
    print(json.dumps({"rows": len(rows), "seen_in_probe": len(seen), "secs": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main(sys.argv[1])
