"""Conditional edge functions for the LangGraph workflow.

Each function inspects the current state and returns the name of the next
node to route to.
"""
from app.core.config import settings
from app.graph.state import GraphState

def route_context_quality(state: GraphState) -> str:
    """Route after context quality check.

    If context is sufficient → generate_answer.
    If context is insufficient and retries remain → rewrite_query (retry).
    If context is insufficient and retries exhausted → handle_insufficient_context.
    """
    context = state.get("context")
    sufficient = context is not None and len(context.selected_chunks) > 0
    if sufficient:
        return "generate_answer"

    retry_count = state.get("retry_count", 0)
    if retry_count < settings.grounding_max_retries:
        return "rewrite_query"
    return "handle_insufficient_context"

def route_grounding(state: GraphState) -> str:
    """Route after grounding validation.

    Supported answers proceed to response preparation.
    Partially supported and unsupported answers are retried
    while retries remain.
    After retries are exhausted, return a safe abstention.
    """
    status = state.get("grounding_status", "unsupported")

    if status == "supported":
        return "prepare_response"

    retry_count = state.get("retry_count", 0)

    if retry_count < settings.grounding_max_retries:
        return "rewrite_query"

    return "handle_unsupported_grounding"

def should_retry(state: GraphState) -> str:
    """Check if we should retry after a failed grounding/insufficient context.

    This is used by the retry increment node to decide whether to loop back
    or go to a terminal node.
    """
    retry_count = state.get("retry_count", 0)
    if retry_count < settings.grounding_max_retries:
        return "retry"
    return "terminal"