"""Vector database for guideline embeddings: FAISS (persisted on disk) with an in-memory numpy fallback.

The 24-d LSA embeddings from core/vectors.py are L2-normalised, so inner product == cosine similarity and a
``faiss.IndexFlatIP`` gives exact nearest-neighbour search. The index is written to ``data/runtime/vector_store/``
(``guidelines.faiss`` + ``guidelines.meta.json``) and re-used on the next start when the fingerprint (SHA-256 of the
corpus text + encoder settings + embeddings) matches; otherwise it is rebuilt. faiss-cpu ships manylinux wheels for
Python 3.13 and 3.14 (tested), so it installs on Streamlit Community Cloud; if the import fails anyway, or
``NG_VECTOR_BACKEND=memory`` is set, the same API runs on numpy (exact cosine, identical scores).

Why FAISS and not Chroma/pgvector: the corpus is 30 excerpts and the server must stay light (no torch, no server
process); FAISS is a real vector index library with persistence, and swapping in pgvector would only change this file.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .config import RUNTIME


def _faiss():
    if os.environ.get("NG_VECTOR_BACKEND", "").lower() == "memory":
        return None
    try:
        import faiss
        return faiss
    except Exception:
        return None


class VectorStore:
    def __init__(self, name: str, ids: list[str], vectors: np.ndarray, fingerprint_extra: str = "",
                 persist_dir: Path | None = None):
        self.name, self.ids = name, list(ids)
        self.vectors = np.ascontiguousarray(vectors, dtype="float32")
        self.dim = int(self.vectors.shape[1])
        self.dir = Path(persist_dir or os.environ.get("NG_VECTOR_DIR", RUNTIME / "vector_store"))
        h = hashlib.sha256()
        h.update(json.dumps(self.ids).encode())
        h.update(fingerprint_extra.encode())
        h.update(np.round(self.vectors, 5).tobytes())
        self.fingerprint = h.hexdigest()
        self.faiss = _faiss()
        self.backend = "faiss" if self.faiss is not None else "memory"
        self.loaded_from_disk = False
        self.index = None
        if self.faiss is not None:
            self.index = self._load() or self._build()

    # -------------------------------------------------------------- persistence
    @property
    def index_path(self) -> Path:
        return self.dir / f"{self.name}.faiss"

    @property
    def meta_path(self) -> Path:
        return self.dir / f"{self.name}.meta.json"

    def _load(self):
        try:
            meta = json.loads(self.meta_path.read_text())
            if meta.get("fingerprint") != self.fingerprint or meta.get("ids") != self.ids:
                return None
            idx = self.faiss.read_index(str(self.index_path))
            if idx.ntotal != len(self.ids) or idx.d != self.dim:
                return None
            self.loaded_from_disk = True
            return idx
        except Exception:
            return None

    def _build(self):
        idx = self.faiss.IndexFlatIP(self.dim)
        idx.add(self.vectors)
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self.index_path.with_suffix(".tmp")
            self.faiss.write_index(idx, str(tmp))
            os.replace(tmp, self.index_path)
            self.meta_path.write_text(json.dumps({"fingerprint": self.fingerprint, "ids": self.ids, "dim": self.dim,
                                                  "metric": "inner_product (cosine on unit vectors)", "backend": "faiss.IndexFlatIP"}))
        except OSError:          # read-only disk: keep the in-memory index
            pass
        return idx

    # -------------------------------------------------------------- search
    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        q = np.ascontiguousarray(np.asarray(query_vec, dtype="float32").reshape(1, -1))
        k = max(1, min(int(k), len(self.ids)))
        if self.index is not None:
            scores, idx = self.index.search(q, k)
            return [(self.ids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]
        sims = self.vectors @ q[0]
        order = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in order]

    def similarities(self, query_vec: np.ndarray) -> np.ndarray:
        """Score for every stored vector, in id order (used by hybrid fusion)."""
        out = np.zeros(len(self.ids), dtype="float64")
        pos = {cid: i for i, cid in enumerate(self.ids)}
        for cid, s in self.search(query_vec, len(self.ids)):
            out[pos[cid]] = s
        return out

    def info(self) -> dict:
        return {"backend": self.backend, "vectors": len(self.ids), "dim": self.dim, "persisted": self.index_path.exists(),
                "loaded_from_disk": self.loaded_from_disk, "path": f"{self.dir.name}/{self.index_path.name}"
                if self.backend == "faiss" else None, "fingerprint": self.fingerprint[:12]}
