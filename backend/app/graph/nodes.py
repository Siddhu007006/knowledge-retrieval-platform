"""LangGraph node functions — one per workflow stage.

Each node takes a ``GraphState`` dict and returns a partial dict to update
the state.  Nodes are kept small and focused on a single responsibility.
"""

import time
from typing import Optional

from app.core.config import settings
from app.core.exceptions import EmptyQueryError, QueryTooLongError
from app.core.logging import log_event
from app.agents.clarification import ClarificationAgent
from app.agents.conversation_memory import ConversationMemoryAgent
from app.agents.query_understanding import QueryUnderstandingAgent
from app.agents.retrieval import RetrievalAgent
from app.agents.response_generation import ResponseGenerationAgent
from app.graph.state import GraphState
from app.models.responses import Citation
from app.models.workflow import (
    ContextPackage,
    GroundingResult,
    RerankedChunk,
    RetrievedCandidate,
    WorkflowEvent,
)

# Agent factories intentionally create request-local helpers. In particular, the
# retrieval helper must reload FAISS metadata so uploads/deletions become visible.
def _get_retrieval_agent() -> RetrievalAgent:
    return RetrievalAgent()


def _get_memory_agent() -> ConversationMemoryAgent:
    return ConversationMemoryAgent()


def _get_query_agent() -> QueryUnderstandingAgent:
    return QueryUnderstandingAgent()


def _get_clarification_agent() -> ClarificationAgent:
    return ClarificationAgent()


def _get_response_agent() -> ResponseGenerationAgent:
    return ResponseGenerationAgent()


def _event(state: GraphState, stage: str, message: str, **data) -> WorkflowEvent:
    return WorkflowEvent(stage=stage, message=message, data=data)


# ---------------------------------------------------------------------------
# Agent nodes: conversation memory and clarification
# ---------------------------------------------------------------------------

def load_conversation(state: GraphState) -> dict:
    conversation_id = state.get("conversation_id", "")
    history = _get_memory_agent().load(conversation_id) if conversation_id else []
    return {
        "conversation_history": history,
        "events": [_event(state, "conversation_memory", "Conversation history loaded", message_count=len(history))],
    }


def clarification_check(state: GraphState) -> dict:
    required, question = _get_clarification_agent().analyze(
        state.get("original_question", ""), state.get("conversation_history", [])
    )
    return {
        "clarification_required": required,
        "clarification_question": question,
        "events": [_event(state, "clarification", "Clarification required" if required else "Query sufficiently specified")],
    }


def handle_clarification(state: GraphState) -> dict:
    answer = state.get("clarification_question") or "Could you clarify what topic or document you are referring to?"
    return {
        "answer": answer,
        "citations": [],
        "grounding_status": "clarification_required",
        "events": [_event(state, "clarification", "Clarification question prepared")],
    }


def save_conversation(state: GraphState) -> dict:
    conversation_id = state.get("conversation_id", "")
    question = state.get("original_question", "").strip()
    answer = state.get("answer", "").strip()
    if conversation_id and question and answer:
        _get_memory_agent().save_exchange(conversation_id, question, answer)
    return {"events": [_event(state, "conversation_memory", "Conversation exchange saved")]}


# ---------------------------------------------------------------------------
# Node: validate_query
# ---------------------------------------------------------------------------


def validate_query(state: GraphState) -> dict:
    question = state.get("original_question", "").strip()
    if not question:
        raise EmptyQueryError()
    if len(question) > settings.max_query_length:
        raise QueryTooLongError(
            f"Query length {len(question)} exceeds maximum {settings.max_query_length}."
        )
    log_event(
        "Query validated",
        request_id=state.get("request_id"),
        stage="validate_query",
    )
    return {
        "retry_count": state.get("retry_count", 0),
        "events": [
            _event(state, "validate_query", "Query validated", length=len(question))
        ],
    }


# ---------------------------------------------------------------------------
# Node: rewrite_query
# ---------------------------------------------------------------------------


def rewrite_query(state: GraphState) -> dict:
    """Rewrite the query for better retrieval, or fall back to the original."""
    original = state["original_question"]
    document_ids = state.get("document_ids")
    retry_count = state.get("retry_count", 0)

    if not settings.query_rewrite_enabled:
        return {
            "rewritten_question": original,
            "rewrite_reason": "Rewriting disabled by configuration.",
            "rewrite_applied": False,
            "events": [_event(state, "rewrite_query", "Rewriting disabled")],
        }

    try:
        result = _get_query_agent().run(
            original, document_ids=document_ids,
            history=state.get("conversation_history", []), retry_count=retry_count
        )
        rewritten = result.rewritten_query or original
        applied = rewritten.strip().lower() != original.strip().lower()
        log_event(
            "Query rewritten" if applied else "Query not rewritten",
            request_id=state.get("request_id"),
            stage="rewrite_query",
        )
        return {
            "rewritten_question": rewritten,
            "rewrite_reason": result.reason,
            "rewrite_applied": applied,
            "events": [
                _event(
                    state,
                    "rewrite_query",
                    "Query rewritten" if applied else "Query not rewritten",
                    original=original,
                    rewritten=rewritten,
                    reason=result.reason,
                )
            ],
        }
    except Exception as exc:
        log_event(
            "Query rewrite failed, using original",
            level="WARNING",
            request_id=state.get("request_id"),
            stage="rewrite_query",
            error=str(exc),
        )
        return {
            "rewritten_question": original,
            "rewrite_reason": f"Rewrite failed, using original query.",
            "rewrite_applied": False,
            "events": [
                _event(
                    state,
                    "rewrite_query",
                    "Rewrite failed, falling back",
                    error=str(exc),
                )
            ],
        }


# ---------------------------------------------------------------------------
# Node: retrieve_candidates
# ---------------------------------------------------------------------------


def retrieve_candidates(state: GraphState) -> dict:
    agent = _get_retrieval_agent()
    query = state.get("rewritten_question") or state["original_question"]
    document_ids = state.get("document_ids")
    start = time.perf_counter()
    candidates = agent.retrieve(query=query, document_ids=document_ids)
    latency_ms = (time.perf_counter() - start) * 1000
    log_event(
        "Retrieval completed",
        request_id=state.get("request_id"),
        stage="retrieve_candidates",
        candidate_count=len(candidates),
        latency_ms=round(latency_ms, 2),
    )
    return {
        "retrieved_candidates": candidates,
        "events": [
            _event(
                state,
                "retrieve_candidates",
                "Retrieval completed",
                candidate_count=len(candidates),
                latency_ms=round(latency_ms, 2),
            )
        ],
    }


# ---------------------------------------------------------------------------
# Node: rerank_candidates
# ---------------------------------------------------------------------------


def rerank_candidates(state: GraphState) -> dict:
    agent = _get_retrieval_agent()
    query = state.get("rewritten_question") or state["original_question"]
    candidates: list[RetrievedCandidate] = state.get("retrieved_candidates", [])
    start = time.perf_counter()
    reranked = agent.rerank(query=query, candidates=candidates)
    latency_ms = (time.perf_counter() - start) * 1000
    log_event(
        "Reranking completed",
        request_id=state.get("request_id"),
        stage="rerank_candidates",
        reranked_count=len(reranked),
        latency_ms=round(latency_ms, 2),
    )
    return {
        "reranked_chunks": reranked,
        "events": [
            _event(
                state,
                "rerank_candidates",
                "Reranking completed",
                reranked_count=len(reranked),
                latency_ms=round(latency_ms, 2),
            )
        ],
    }


# ---------------------------------------------------------------------------
# Node: filter_context
# ---------------------------------------------------------------------------


def filter_context(state: GraphState) -> dict:
    agent = _get_retrieval_agent()
    reranked: list[RerankedChunk] = state.get("reranked_chunks", [])
    question = state["original_question"]
    rewritten = state.get("rewritten_question") or question
    document_ids = state.get("document_ids")
    context = agent.build_context(question, rewritten, reranked, document_ids)
    log_event(
        "Context filtered",
        request_id=state.get("request_id"),
        stage="filter_context",
        selected_count=len(context.selected_chunks),
    )
    return {
        "context": context,
        "events": [
            _event(
                state,
                "filter_context",
                "Context filtered",
                selected_count=len(context.selected_chunks),
            )
        ],
    }


# ---------------------------------------------------------------------------
# Node: check_context_quality
# ---------------------------------------------------------------------------


def check_context_quality(state: GraphState) -> dict:
    """Check if the context has enough evidence. The routing is handled by edges."""
    context = state.get("context")
    sufficient = context is not None and len(context.selected_chunks) > 0
    log_event(
        "Context quality checked",
        request_id=state.get("request_id"),
        stage="check_context_quality",
        sufficient=sufficient,
    )
    return {
        "events": [
            _event(
                state,
                "check_context_quality",
                "Context quality checked",
                sufficient=sufficient,
                selected_count=len(context.selected_chunks) if context else 0,
            )
        ],
    }


# ---------------------------------------------------------------------------
# Node: generate_answer
# ---------------------------------------------------------------------------


def generate_answer_node(state: GraphState) -> dict:
    context: ContextPackage = state["context"]
    start = time.perf_counter()
    answer = _get_response_agent().generate(context)
    latency_ms = (time.perf_counter() - start) * 1000
    log_event(
        "Answer generated",
        request_id=state.get("request_id"),
        stage="generate_answer",
        latency_ms=round(latency_ms, 2),
    )
    return {
        "answer": answer,
        "events": [
            _event(
                state,
                "generate_answer",
                "Answer generated",
                latency_ms=round(latency_ms, 2),
            )
        ],
    }


# ---------------------------------------------------------------------------
# Node: check_grounding
# ---------------------------------------------------------------------------


def check_grounding_node(state: GraphState) -> dict:
    context: ContextPackage = state["context"]
    answer = state.get("answer", "")
    start = time.perf_counter()
    grounding_result = _get_response_agent().verify(context, answer)
    latency_ms = (time.perf_counter() - start) * 1000
    log_event(
        "Grounding checked",
        request_id=state.get("request_id"),
        stage="check_grounding",
        grounding_status=grounding_result.status,
        latency_ms=round(latency_ms, 2),
    )
    return {
        "grounding_result": grounding_result,
        "grounding_status": grounding_result.status,
        "events": [
            _event(
                state,
                "check_grounding",
                "Grounding checked",
                status=grounding_result.status,
                latency_ms=round(latency_ms, 2),
            )
        ],
    }


# ---------------------------------------------------------------------------
# Node: prepare_response
# ---------------------------------------------------------------------------


def prepare_response(state: GraphState) -> dict:
    context: ContextPackage = state["context"]
    citations: list[Citation] = []
    for item in context.selected_chunks:
        excerpt = item.chunk.text[:300]
        if len(item.chunk.text) > 300:
            excerpt += "..."
        citations.append(
            Citation(
                document_id=item.chunk.document_id,
                filename=item.chunk.filename,
                page_number=item.chunk.page_number,
                section=item.chunk.section,
                chunk_id=item.chunk.chunk_id,
                excerpt=excerpt,
            )
        )
    log_event(
        "Response prepared",
        request_id=state.get("request_id"),
        stage="prepare_response",
        citation_count=len(citations),
    )
    return {
        "citations": citations,
        "events": [
            _event(
                state,
                "prepare_response",
                "Response prepared",
                citation_count=len(citations),
            )
        ],
    }


# ---------------------------------------------------------------------------
# Node: handle_insufficient_context
# ---------------------------------------------------------------------------


def handle_insufficient_context(state: GraphState) -> dict:
    """Terminal node for when context is insufficient after all retries."""
    log_event(
        "Insufficient context — returning safe response",
        request_id=state.get("request_id"),
        stage="handle_insufficient_context",
        retry_count=state.get("retry_count", 0),
    )
    answer = (
        "I don't have enough information from the provided documents to answer "
        "this question. Please try uploading relevant documents or rephrasing "
        "your question."
    )
    return {
        "answer": answer,
        "grounding_status": "insufficient_context",
        "grounding_result": GroundingResult(
            status="unsupported",
            reasons=["No sufficient context was found after retrieval and reranking."],
        ),
        "citations": [],
        "events": [
            _event(
                state,
                "handle_insufficient_context",
                "Insufficient context response prepared",
                retry_count=state.get("retry_count", 0),
            )
        ],
    }


# ---------------------------------------------------------------------------
# Node: handle_unsupported_grounding
# ---------------------------------------------------------------------------

def handle_unsupported_grounding(state: GraphState) -> dict:
    """Return a safe abstention when grounding cannot be established."""

    log_event(
        "Unsupported grounding — returning safe abstention",
        request_id=state.get("request_id"),
        stage="handle_unsupported_grounding",
        retry_count=state.get("retry_count", 0),
    )

    previous_result: GroundingResult | None = state.get("grounding_result")
    reasons = previous_result.reasons if previous_result else []

    safe_result = GroundingResult(
        status="unsupported",
        reasons=reasons or [
            "The generated answer could not be fully supported by retrieved evidence."
        ],
        supported_claims=[],
        unsupported_claims=(
            previous_result.unsupported_claims if previous_result else []
        ),
    )

    return {
        "answer": (
            "I couldn't verify a sufficiently supported answer from the "
            "provided documents. Please upload relevant documents or "
            "rephrase your question."
        ),
        "grounding_status": "insufficient_context",
        "grounding_result": safe_result,
        "citations": [],
        "events": [
            _event(
                state,
                "handle_unsupported_grounding",
                "Safe abstention response prepared",
                retry_count=state.get("retry_count", 0),
            )
        ],
    }