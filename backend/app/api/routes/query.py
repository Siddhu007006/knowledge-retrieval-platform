"""Knowledge-base query endpoint."""
from fastapi import APIRouter

from app.core.logging import log_event, new_request_id
from app.graph.workflow import build_workflow
from app.models.requests import QueryRequest
from app.models.responses import QueryResponse, RetrievalMetadata

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    request_id = new_request_id()
    log_event("Query received", request_id=request_id, question=req.question)

    document_ids = None
    if req.document_id:
        document_ids = [req.document_id]
    elif req.document_ids:
        document_ids = req.document_ids

    state = build_workflow().run(
        question=req.question,
        document_ids=document_ids,
        request_id=request_id,
        conversation_id=req.conversation_id,
    )
    context = state.get("context")
    metadata = RetrievalMetadata(
        candidate_count=len(state.get("retrieved_candidates", [])),
        reranked_count=len(state.get("reranked_chunks", [])),
        selected_count=len(context.selected_chunks) if context else 0,
        rewrite_applied=state.get("rewrite_applied", False),
        rewritten_query=state.get("rewritten_question"),
        rewrite_reason=state.get("rewrite_reason"),
        retry_count=state.get("retry_count", 0),
    )
    return QueryResponse(
        conversation_id=state.get("conversation_id"),
        clarification_required=state.get("clarification_required", False),
        answer=state.get("answer", ""),
        citations=state.get("citations", []),
        grounding_status=state.get("grounding_status", "unsupported"),
        retrieval_metadata=metadata,
    )
