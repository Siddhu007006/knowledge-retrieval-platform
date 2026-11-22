
"""Score-preserving reranking using the original FAISS similarity scores."""

from typing import List

from app.core.config import settings
from app.models.workflow import RerankedChunk, RetrievedCandidate


class Reranker:
    """
    Preserve the original FAISS ranking.

    This implementation intentionally does not perform additional
    embedding-based reranking because the existing approach degraded
    retrieval accuracy in the current evaluation dataset.
    """

    def __init__(self, top_k: int | None = None) -> None:
        self.top_k = top_k if top_k is not None else settings.rerank_top_k

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedCandidate],
    ) -> List[RerankedChunk]:

        if not candidates:
            return []

        # Preserve relevance ordering using the original FAISS scores.
        ordered = sorted(
            candidates,
            key=lambda candidate: candidate.score,
            reverse=True,
        )

        return [
            RerankedChunk(
                chunk=candidate.chunk,
                score=candidate.score,
                rerank_score=float(candidate.score),
            )
            for candidate in ordered[:self.top_k]
        ]
