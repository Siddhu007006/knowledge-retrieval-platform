"""Initial retrieval stage: query embedding → FAISS search → candidate pool."""

from typing import List, Optional

from app.core.config import settings
from app.models.workflow import RetrievedCandidate
from app.retrieval.vector_store import VectorStore


class Retriever:
    """Performs initial vector retrieval with configurable ``k``."""

    def __init__(self, vector_store: Optional[VectorStore] = None) -> None:
        self.vector_store = vector_store or VectorStore()

    def retrieve(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        k: Optional[int] = None,
    ) -> List[RetrievedCandidate]:
        k = k or settings.initial_retrieval_k
        results = self.vector_store.search(query, k=k, document_ids=document_ids)
        return [
            RetrievedCandidate(chunk=chunk, score=score) for chunk, score in results
        ]
