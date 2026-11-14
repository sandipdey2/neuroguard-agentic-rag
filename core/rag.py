"""Retrieval over 30 verbatim excerpts from CDC and NHLBI public pages (Project 6 lineage, extended).

retrieve -> inject -> generate (Lewis et al., 2020): ``search`` returns ranked chunks with id, title, publisher
and URL; the Evidence agent injects them into the writer's context, and the writer must cite ids like [G-03].
Hybrid retrieval (default): score = 0.6 * sparse TF-IDF cosine + 0.4 * dense LSA cosine (core/vectors.py) plus a small
topic boost when the query names a chunk's topic. Dense scores come from the FAISS vector store
(core/vector_store.py; numpy fallback). mode="sparse" or "dense" gives either component alone (used by the
evaluation to compare them). ``boosts`` lets the learning loop add a bounded per-chunk bonus (capped at 0.06).
"""
from __future__ import annotations

from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import load_guidelines
from .vector_store import VectorStore
from .vectors import DenseEncoder

TOPIC_SYNONYMS = {
    "hypertension": ["hypertension", "blood pressure"], "glucose": ["glucose", "diabetes", "blood sugar"],
    "bmi": ["bmi", "obesity", "weight"], "smoking": ["smok", "tobacco", "cigarette"], "age": ["age", "older"],
    "heart_disease": ["heart disease", "atrial fibrillation", "heart"], "emergency": ["emergency", "symptom", "fast", "911"],
    "tia": ["tia", "mini-stroke", "transient"], "lifestyle": ["lifestyle", "activity", "diet", "alcohol", "prevent"],
}
TOPIC_BOOST = 0.08
MIN_SCORE = 0.04
W_SPARSE, W_DENSE = 0.6, 0.4
MAX_LEARNED_BOOST = 0.06


class GuidelineIndex:
    def __init__(self):
        self.docs = load_guidelines()
        corpus = [f"{d['title']}. {d['text']}" for d in self.docs]
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(corpus)
        self.by_id = {d["id"]: d for d in self.docs}
        self.dense = DenseEncoder(corpus)
        # dense vectors live in a real vector index (FAISS, persisted; numpy fallback) - core/vector_store.py
        self.store = VectorStore("guidelines", [d["id"] for d in self.docs], self.dense.doc_vectors,
                                 fingerprint_extra=f"lsa{self.dense.dim}-seed42|" + "|".join(corpus))

    def search(self, query: str, k: int = 4, mode: str = "hybrid", boosts: dict | None = None) -> list[dict]:
        sparse = cosine_similarity(self.vec.transform([query]), self.matrix).ravel()
        dense = self.store.similarities(self.dense.encode([query])[0]).clip(min=0)
        scores = {"sparse": sparse, "dense": dense}.get(mode, W_SPARSE * sparse + W_DENSE * dense).copy()
        ql = query.lower()
        for i, d in enumerate(self.docs):
            if any(s in ql for s in TOPIC_SYNONYMS.get(d["topic"], [])):
                scores[i] += TOPIC_BOOST
            if boosts:
                scores[i] += min(MAX_LEARNED_BOOST, max(0.0, float(boosts.get(d["id"], 0.0))))
        out = []
        for i in scores.argsort()[::-1][:k]:
            if scores[i] < MIN_SCORE:
                continue
            d = self.docs[i]
            out.append({"id": d["id"], "title": d["title"], "publisher": d["publisher"], "url": d["url"],
                        "text": d["text"], "score": round(float(scores[i]), 3),
                        "sparse": round(float(sparse[i]), 3), "dense": round(float(dense[i]), 3), "mode": mode})
        return out


@lru_cache(maxsize=1)
def get_index() -> GuidelineIndex:
    return GuidelineIndex()
