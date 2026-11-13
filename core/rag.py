"""Retrieval over 30 verbatim excerpts from CDC and NHLBI public pages (Project 6 lineage, extended).

retrieve -> inject -> generate (Lewis et al., 2020): ``search`` returns ranked chunks with id, title, publisher
and URL; the Evidence agent injects them into the writer's context, and the writer must cite ids like [G-03].
TF-IDF (unigrams + bigrams, sublinear tf) with a small topic boost when the query names a chunk's topic.
"""
from __future__ import annotations

from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import load_guidelines

TOPIC_SYNONYMS = {
    "hypertension": ["hypertension", "blood pressure"], "glucose": ["glucose", "diabetes", "blood sugar"],
    "bmi": ["bmi", "obesity", "weight"], "smoking": ["smok", "tobacco", "cigarette"], "age": ["age", "older"],
    "heart_disease": ["heart disease", "atrial fibrillation", "heart"], "emergency": ["emergency", "symptom", "fast", "911"],
    "tia": ["tia", "mini-stroke", "transient"], "lifestyle": ["lifestyle", "activity", "diet", "alcohol", "prevent"],
}
TOPIC_BOOST = 0.08
MIN_SCORE = 0.04


class GuidelineIndex:
    def __init__(self):
        self.docs = load_guidelines()
        corpus = [f"{d['title']}. {d['text']}" for d in self.docs]
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(corpus)
        self.by_id = {d["id"]: d for d in self.docs}

    def search(self, query: str, k: int = 4) -> list[dict]:
        scores = cosine_similarity(self.vec.transform([query]), self.matrix).ravel()
        ql = query.lower()
        for i, d in enumerate(self.docs):
            if any(s in ql for s in TOPIC_SYNONYMS.get(d["topic"], [])):
                scores[i] += TOPIC_BOOST
        out = []
        for i in scores.argsort()[::-1][:k]:
            if scores[i] < MIN_SCORE:
                continue
            d = self.docs[i]
            out.append({"id": d["id"], "title": d["title"], "publisher": d["publisher"], "url": d["url"],
                        "text": d["text"], "score": round(float(scores[i]), 3)})
        return out


@lru_cache(maxsize=1)
def get_index() -> GuidelineIndex:
    return GuidelineIndex()
