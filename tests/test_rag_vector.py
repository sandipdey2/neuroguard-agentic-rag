"""Vector store (FAISS, persisted; numpy fallback) and hybrid retrieval that uses it."""
import numpy as np

from core.vector_store import VectorStore


def _vecs(n=12, d=8, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, d)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_faiss_matches_numpy_and_persists(tmp_path, monkeypatch):
    v = _vecs()
    ids = [f"G-{i:02d}" for i in range(len(v))]
    s1 = VectorStore("t", ids, v, persist_dir=tmp_path)
    assert s1.backend == "faiss" and s1.index_path.exists() and not s1.loaded_from_disk
    s2 = VectorStore("t", ids, v, persist_dir=tmp_path)
    assert s2.loaded_from_disk
    monkeypatch.setenv("NG_VECTOR_BACKEND", "memory")
    s3 = VectorStore("t", ids, v, persist_dir=tmp_path)
    assert s3.backend == "memory"
    q = v[3]
    assert s1.search(q, 3)[0][0] == "G-03" == s3.search(q, 3)[0][0]
    assert np.allclose(s1.similarities(q), s3.similarities(q), atol=1e-5)


def test_fingerprint_change_rebuilds(tmp_path):
    v = _vecs()
    ids = [str(i) for i in range(len(v))]
    VectorStore("t", ids, v, persist_dir=tmp_path)
    s = VectorStore("t", ids, v, fingerprint_extra="new corpus", persist_dir=tmp_path)
    assert not s.loaded_from_disk


def test_guideline_search_uses_vector_store():
    from core.rag import get_index
    idx = get_index()
    assert idx.store.info()["vectors"] == 30
    top = idx.search("high blood pressure and stroke", k=3)
    assert top[0]["id"] == "G-03"
    assert idx.search("brain attack warning signs face drooping", k=3, mode="dense")[0]["dense"] > 0
