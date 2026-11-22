"""Retrieval Agent: semantic search, reranking, and evidence-context selection."""
from typing import Optional

from app.models.workflow import ContextPackage, RetrievedCandidate, RerankedChunk
from app.retrieval.context_filter import ContextFilter
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import Retriever


class RetrievalAgent:
    def __init__(self, retriever: Retriever | None = None) -> None:
        self._retriever: Retriever | None = retriever
        self.reranker = Reranker()
        self.context_filter = ContextFilter()

    def retrieve(self, query: str, document_ids: Optional[list[str]] = None, k: int | None = None) -> list[RetrievedCandidate]:
        # Load the on-disk FAISS metadata on demand so new uploads/deletions are visible.
        if self._retriever is None:
            self._retriever = Retriever()
        return self._retriever.retrieve(query=query, document_ids=document_ids, k=k)

    def rerank(self, query: str, candidates: list[RetrievedCandidate]) -> list[RerankedChunk]:
        return self.reranker.rerank(query=query, candidates=candidates)

    def build_context(self, question: str, rewritten_query: str, reranked: list[RerankedChunk],
                      document_ids: Optional[list[str]] = None) -> ContextPackage:
        return self.context_filter.filter(
            question=question,
            rewritten_query=rewritten_query,
            reranked=reranked,
            document_ids=document_ids,
        )
