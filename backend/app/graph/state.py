"""Typed graph state for the LangGraph workflow.

Uses ``TypedDict`` with ``Annotated`` reducers for list fields that
accumulate across nodes.
"""

import operator
from typing import Annotated, List, Optional

from typing_extensions import TypedDict

from app.models.workflow import (
    ContextPackage,
    GroundingResult,
    RerankedChunk,
    RetrievedCandidate,
    WorkflowEvent,
)


class GraphState(TypedDict, total=False):
    """State object passed between graph nodes.

    Fields:
      original_question: the user's original question (never modified).
      rewritten_question: the query used for retrieval (may equal original).
      rewrite_reason: why the query was (or was not) rewritten.
      rewrite_applied: whether rewriting changed the query.
      document_ids: optional scope for document-filtered retrieval.
      retrieved_candidates: candidates from initial vector retrieval.
      reranked_chunks: candidates after reranking.
      context: the final filtered context package.
      answer: the generated answer text.
      citations: source citations for the answer.
      grounding_status: supported | partially_supported | unsupported | insufficient_context.
      grounding_result: detailed grounding check output.
      retry_count: how many retrieval/regeneration retries have occurred.
      events: accumulated workflow-stage events.
      error: error message if the workflow failed.
      request_id: correlation ID for logging.
    """

    original_question: str
    conversation_id: str
    conversation_history: list[dict]
    clarification_required: bool
    clarification_question: Optional[str]
    rewritten_question: str
    rewrite_reason: str
    rewrite_applied: bool
    document_ids: Optional[List[str]]
    retrieved_candidates: List[RetrievedCandidate]
    reranked_chunks: List[RerankedChunk]
    context: Optional[ContextPackage]
    answer: str
    citations: list
    grounding_status: str
    grounding_result: Optional[GroundingResult]
    retry_count: int
    events: Annotated[List[WorkflowEvent], operator.add]
    error: Optional[str]
    request_id: str
