"""Similar labelled examples for the LLM prompt: word + char TF-IDF over train contexts, cosine top 8 with at most one
row per conversation. Same text, settings and ranking as experiments/llm_tail/build_inputs.py, so the service
picks the examples the LLM test used. The saved file also holds each train row's prompt payload, so the
service never loads train at runtime.

  python scripts/build_retrieval.py      # writes models/retrieval.joblib
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .data import ROOT
from .prompts import MAX_TURN_CHARS

DEFAULT_PATH = ROOT / "models" / "retrieval.joblib"


def text(r: dict) -> str:
    return "\n".join(f"{t['speaker']}: {(t.get('text') or '')[:MAX_TURN_CHARS]}" for t in r["context"])


def payload(r: dict) -> dict:
    """What `prompts._example_block` reads from a train row."""
    return {"id": r["id"], "intent": r["intent"], "action": r["action"],
            "context": [{k: t[k] for k in ("speaker", "text") if k in t} for t in r["context"]]}


class Retriever:
    """`build(train_rows)` or `load(path)`; then `top_k(row)` -> [(train id, cosine)], best first, and
    `examples(row)` -> the payloads of those rows. Read-only after construction, so safe across threads."""

    def __init__(self, vw, vc, by_feature: sp.csr_matrix, payloads: list[dict], meta: dict | None = None):
        self.vw, self.vc = vw, vc
        self.by_feature = by_feature          # features x train rows, columns L2-normalised (train matrix, transposed)
        self.payloads = payloads
        self.meta = meta or {}
        self.ids = [p["id"] for p in payloads]
        self._conv = [i.rsplit("-", 1)[0] for i in self.ids]

    @classmethod
    def build(cls, rows: list[dict]) -> "Retriever":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200000)
        m = normalize(sp.hstack([vw.fit_transform([text(r) for r in rows]),
                                 vc.fit_transform([text(r) for r in rows])]).tocsr())
        for v in (vw, vc):
            if hasattr(v, "stop_words_"):
                del v.stop_words_                                     # introspection only; keeps the file small
        # q @ M.T is how the experiment scored; keeping M.T as csr gives the same sums in the same order
        return cls(vw, vc, m.T.tocsr(), [payload(r) for r in rows], {"rows": len(rows), "features": m.shape[1]})

    def save(self, path: Path = DEFAULT_PATH) -> None:
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vw": self.vw, "vc": self.vc, "by_feature": self.by_feature, "payloads": self.payloads,
                     "meta": self.meta}, path, compress=3)

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "Retriever":
        import joblib
        b = joblib.load(path)
        return cls(b["vw"], b["vc"], b["by_feature"], b["payloads"], b.get("meta"))

    def scores(self, row: dict) -> np.ndarray:
        from sklearn.preprocessing import normalize
        s = text(row)
        q = normalize(sp.hstack([self.vw.transform([s]), self.vc.transform([s])]).tocsr())
        return (q @ self.by_feature).toarray()[0]

    def _top(self, row: dict, k: int) -> list[tuple[int, float]]:
        s = self.scores(row)
        picked, convs = [], set()
        for j in np.argsort(-s):
            if self._conv[j] in convs:
                continue
            convs.add(self._conv[j])
            picked.append((int(j), float(s[j])))
            if len(picked) == k:
                break
        return picked

    def top_k(self, row: dict, k: int = 8) -> list[tuple[str, float]]:
        return [(self.ids[j], s) for j, s in self._top(row, k)]

    def examples(self, row: dict, k: int = 8) -> list[dict]:
        return [self.payloads[j] for j, _ in self._top(row, k)]
