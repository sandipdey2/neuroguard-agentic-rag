"""Embeddings & vector search, CPU-only (no torch): Latent Semantic Analysis dense vectors.

TF-IDF (unigram + bigram, sublinear) -> TruncatedSVD (24 components, seed 42) -> L2 normalisation gives each
guideline excerpt a dense 24-dimensional vector; queries are projected into the same space and ranked by cosine
similarity (= dot product of unit vectors). LSA captures co-occurrence ("blood sugar" ~ "diabetes" ~ "glucose")
that pure keyword matching misses. It runs inside the MCP server, which deliberately has no API key, so it cannot
call a hosted embedding model; evaluation.ipynb compares it with OpenAI text-embedding-3-small through the
Vocareum endpoint (which does support embeddings) on the same labelled queries.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

N_COMPONENTS = 24
SEED = 42


class DenseEncoder:
    def __init__(self, corpus: list[str], n_components: int = N_COMPONENTS):
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        x = self.tfidf.fit_transform(corpus)
        self.svd = TruncatedSVD(n_components=min(n_components, x.shape[1] - 1, x.shape[0] - 1), random_state=SEED)
        self.doc_vectors = normalize(self.svd.fit_transform(x))
        self.dim = self.doc_vectors.shape[1]
        self.explained_variance = float(self.svd.explained_variance_ratio_.sum())

    def encode(self, texts: list[str]) -> np.ndarray:
        return normalize(self.svd.transform(self.tfidf.transform(texts)))

    def similarities(self, query: str) -> np.ndarray:
        return self.doc_vectors @ self.encode([query])[0]
