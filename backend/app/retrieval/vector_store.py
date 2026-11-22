"""High-level vector store facade combining embeddings + FAISS index + document store."""

from typing import List, Optional

import numpy as np

from app.core.config import settings
from app.models.workflow import ChunkModel
from app.retrieval.embeddings import embed_documents, embed_query
from app.storage.documents import DocumentStore
from app.storage.indexes import VectorIndex


class VectorStore:
    """Orchestrates embedding generation, FAISS indexing, and metadata lookup."""

    def __init__(
        self,
        document_store: Optional[DocumentStore] = None,
        vector_index: Optional[VectorIndex] = None,
    ) -> None:
        self.documents = document_store or DocumentStore()
        self.index = vector_index or VectorIndex()
        self._dim = self.index.dimension

    def add_chunks(self, chunks: List[ChunkModel]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors = embed_documents(texts)
        if self._dim == 0:
            self._dim = vectors.shape[-1]
        self.index.add(
            vectors=vectors,
            chunk_ids=[c.chunk_id for c in chunks],
            document_ids=[c.document_id for c in chunks],
        )

    def search(
        self,
        query: str,
        k: int,
        document_ids: Optional[List[str]] = None,
    ) -> List[tuple[ChunkModel, float]]:
        query_vec = embed_query(query)
        raw = self.index.search(query_vec, k=k, document_ids=document_ids)
        results: List[tuple[ChunkModel, float]] = []
        for chunk_id, score in raw:
            chunk = self.documents.get_chunk(chunk_id)
            if chunk is not None:
                results.append((chunk, score))
        return results

    def remove_document(self, document_id: str) -> int:
        removed = self.index.remove_document(document_id)
        return removed

    def total_vectors(self) -> int:
        return self.index.total()
