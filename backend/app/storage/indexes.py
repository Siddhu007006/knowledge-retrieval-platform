"""FAISS vector index manager with persistence.

Maintains a single FAISS ``IndexFlatIP`` (inner-product) over L2-normalized
vectors so that inner product equals cosine similarity.  A parallel
metadata list maps each FAISS position to a ``chunk_id`` and
``document_id`` for document-scoped filtering and deletion.

The index is persisted to disk (``index.faiss`` + ``meta.json``) so it
survives process restarts.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from app.core.config import settings


class VectorIndex:
    """FAISS index with metadata mapping and disk persistence."""

    def __init__(self, base_dir: Optional[Path] = None, dim: int = 0) -> None:
        self._dir = base_dir or settings.index_path()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self._dir / "index.faiss"
        self._meta_file = self._dir / "meta.json"
        self._dim = dim
        self._index: Optional[faiss.IndexFlatIP] = None
        self._meta: List[dict] = []
        self._load()

    def _load(self) -> None:
        if self._index_file.exists() and self._meta_file.exists():
            self._index = faiss.read_index(str(self._index_file))
            self._meta = json.loads(self._meta_file.read_text())
            self._dim = self._index.d
        else:
            self._meta = []

    def _save(self) -> None:
        if self._index is not None:
            faiss.write_index(self._index, str(self._index_file))
        self._meta_file.write_text(json.dumps(self._meta, ensure_ascii=False))

    def _ensure_index(self, dim: int) -> None:
        if self._index is None:
            self._dim = dim
            self._index = faiss.IndexFlatIP(dim)
        elif self._index.d != dim:
            raise ValueError(
                f"Embedding dimension mismatch: index has d={self._index.d}, got d={dim}"
            )

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return (vectors / norms).astype(np.float32)

    def add(
        self,
        vectors: np.ndarray,
        chunk_ids: List[str],
        document_ids: List[str],
    ) -> None:
        if len(vectors) == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        self._ensure_index(vectors.shape[1])
        normalized = self._normalize(vectors)
        self._index.add(normalized)
        for cid, did in zip(chunk_ids, document_ids):
            self._meta.append({"chunk_id": cid, "document_id": did})
        self._save()

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        document_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Return ``[(chunk_id, score), ...]`` sorted by descending score.

        When ``document_ids`` is provided, results are filtered to only
        those chunks belonging to the specified documents.  More candidates
        are fetched internally so the post-filter ``k`` is still respected.
        """
        if self._index is None or self._index.ntotal == 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        query = self._normalize(query)
        fetch_k = k if document_ids is None else max(k * 5, k)
        fetch_k = min(fetch_k, self._index.ntotal)
        scores, indices = self._index.search(query, fetch_k)
        results: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._meta):
                continue
            meta = self._meta[idx]
            if document_ids is not None and meta["document_id"] not in document_ids:
                continue
            results.append((meta["chunk_id"], float(score)))
            if len(results) >= k:
                break
        return results

    def remove_document(self, document_id: str) -> int:
        """Remove all vectors belonging to ``document_id`` and rebuild index."""
        if self._index is None:
            return 0
        keep_positions = [
            i for i, m in enumerate(self._meta) if m["document_id"] != document_id
        ]
        removed = len(self._meta) - len(keep_positions)
        if removed == 0:
            return 0
        if not keep_positions:
            self._index = faiss.IndexFlatIP(self._dim) if self._dim else None
            self._meta = []
        else:
            old_vectors = faiss.rev_swig_ptr(
                self._index.get_xb(), self._index.ntotal * self._index.d
            ).reshape(self._index.ntotal, self._index.d)
            kept_vectors = old_vectors[keep_positions].copy()
            self._index = faiss.IndexFlatIP(self._dim)
            self._index.add(kept_vectors)
            self._meta = [self._meta[i] for i in keep_positions]
        self._save()
        return removed

    def total(self) -> int:
        return self._index.ntotal if self._index else 0

    @property
    def dimension(self) -> int:
        return self._dim
