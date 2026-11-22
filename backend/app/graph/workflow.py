"""Milestone 1 LangGraph orchestration for the five knowledge-retrieval agents."""
from typing import Optional
from langgraph.graph import END, START, StateGraph

from app.core.config import settings
from app.core.logging import log_event
from app.graph.edges import route_context_quality, route_grounding
from app.graph.nodes import (
    check_context_quality,
    check_grounding_node,
    clarification_check,
    filter_context,
    generate_answer_node,
    handle_clarification,
    handle_insufficient_context,
    handle_unsupported_grounding,
    load_conversation,
    prepare_response,
    rerank_candidates,
    retrieve_candidates,
    rewrite_query,
    save_conversation,
    validate_query,
)
from app.graph.state import GraphState


def _increment_retry(state: GraphState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


def build_graph():
    """Create the orchestration graph and connect its conditional paths."""
    graph = StateGraph(GraphState)
    graph.add_node("load_conversation", load_conversation)
    graph.add_node("validate_query", validate_query)
    graph.add_node("clarification_check", clarification_check)
    graph.add_node("handle_clarification", handle_clarification)
    graph.add_node("query_understanding", rewrite_query)
    graph.add_node("retrieve_candidates", retrieve_candidates)
    graph.add_node("rerank_candidates", rerank_candidates)
    graph.add_node("filter_context", filter_context)
    graph.add_node("check_context_quality", check_context_quality)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("check_grounding", check_grounding_node)
    graph.add_node("prepare_response", prepare_response)
    graph.add_node("handle_insufficient_context", handle_insufficient_context)
    graph.add_node("handle_unsupported_grounding", handle_unsupported_grounding)
    graph.add_node("increment_retry", _increment_retry)
    graph.add_node("save_conversation", save_conversation)

    graph.add_edge(START, "load_conversation")
    graph.add_edge("load_conversation", "validate_query")
    graph.add_edge("validate_query", "clarification_check")
    graph.add_conditional_edges(
        "clarification_check",
        lambda state: "clarify" if state.get("clarification_required") else "understand",
        {"clarify": "handle_clarification", "understand": "query_understanding"},
    )
    graph.add_edge("query_understanding", "retrieve_candidates")
    graph.add_edge("retrieve_candidates", "rerank_candidates")
    graph.add_edge("rerank_candidates", "filter_context")
    graph.add_edge("filter_context", "check_context_quality")

    graph.add_conditional_edges(
        "check_context_quality",
        route_context_quality,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "increment_retry",
            "handle_insufficient_context": "handle_insufficient_context",
        },
    )
    graph.add_edge("increment_retry", "query_understanding")
    graph.add_edge("generate_answer", "check_grounding")
    graph.add_conditional_edges(
        "check_grounding",
        route_grounding,
        {
            "prepare_response": "prepare_response",
            "rewrite_query": "increment_retry",
            "handle_unsupported_grounding": "handle_unsupported_grounding",
        },
    )

    graph.add_edge("prepare_response", "save_conversation")
    graph.add_edge("handle_clarification", "save_conversation")
    graph.add_edge("handle_insufficient_context", "save_conversation")
    graph.add_edge("handle_unsupported_grounding", "save_conversation")
    graph.add_edge("save_conversation", END)
    return graph.compile()


class QueryWorkflow:
    """Execute one query through the milestone-defined agent workflow."""

    def __init__(self) -> None:
        self._compiled = build_graph()

    def run(
        self,
        question: str,
        document_ids: Optional[list[str]] = None,
        request_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> GraphState:
        import uuid

        rid = request_id or str(uuid.uuid4())
        cid = conversation_id or str(uuid.uuid4())
        initial_state: GraphState = {
            "original_question": question,
            "conversation_id": cid,
            "document_ids": document_ids,
            "retry_count": 0,
            "events": [],
            "request_id": rid,
        }
        log_event("Workflow started", request_id=rid, stage="workflow")
        final_state = self._compiled.invoke(initial_state)
        log_event("Workflow completed", request_id=rid, stage="workflow")
        return final_state


def build_workflow() -> QueryWorkflow:
    return QueryWorkflow()
