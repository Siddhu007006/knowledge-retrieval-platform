"""Query Understanding Agent: resolve conversational context and refine retrieval queries."""
from app.generation.query_rewriter import RewriteResult, rewrite


class QueryUnderstandingAgent:
    def run(self, question: str, document_ids: list[str] | None = None,
            history: list[dict[str, str]] | None = None, retry_count: int = 0) -> RewriteResult:
        return rewrite(
            question,
            document_ids=document_ids,
            retry_count=retry_count,
            conversation_history=history or [],
        )
