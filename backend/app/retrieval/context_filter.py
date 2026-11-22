"""Context filtering: deduplication, weak-candidate removal, scope enforcement.

Runs after reranking, before generation.  Produces the final
``ContextPackage`` that the generator consumes.
"""

from typing import List, Optional

from app.core.config import settings
from app.models.workflow import ContextPackage, RerankedChunk


class ContextFilter:
    """Filters reranked chunks into the final context package."""

    def __init__(self, min_score: Optional[float] = None) -> None:
        self.min_score = min_score if min_score is not None else settings.min_retrieval_score

    def filter(
        self,
        question: str,
        rewritten_query: str,
        reranked: List[RerankedChunk],
        document_ids: Optional[List[str]] = None,
    ) -> ContextPackage:
        seen: set[str] = set()
        selected: List[RerankedChunk] = []

        for item in reranked:
            text_key = item.chunk.text.strip()
            if text_key in seen:
                continue
            if item.rerank_score < self.min_score:
                continue
            if document_ids is not None and item.chunk.document_id not in document_ids:
                continue
            seen.add(text_key)
            selected.append(item)

        source_metadata = [
            {
                "document_id": item.chunk.document_id,
                "filename": item.chunk.filename,
                "page_number": item.chunk.page_number,
                "section": item.chunk.section,
                "chunk_id": item.chunk.chunk_id,
                "score": item.rerank_score,
            }
            for item in selected
        ]

        return ContextPackage(
            question=question,
            rewritten_query=rewritten_query,
            selected_chunks=selected,
            source_metadata=source_metadata,
        )

    def is_sufficient(self, context: ContextPackage) -> bool:
        """Return True if the context has enough evidence for generation."""
        return len(context.selected_chunks) > 0
