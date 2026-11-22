"""Focused tests for the milestone's agent responsibilities."""
from app.agents.clarification import ClarificationAgent
from app.agents.conversation_memory import ConversationMemoryAgent
from app.evaluation.retrieval import evaluate_retrieval, find_relevant_chunk_ids, hit_at_k
from app.models.workflow import ChunkModel


def test_clarification_agent_detects_unresolved_reference_without_history():
    required, prompt = ClarificationAgent().analyze("What about it?", [])
    assert required is True
    assert prompt


def test_clarification_agent_uses_history_as_antecedent():
    history = [{"role": "user", "content": "Explain the VPN setup."}]
    required, prompt = ClarificationAgent().analyze("What about it?", history)
    assert required is False
    assert prompt is None


def test_conversation_memory_persists_and_limits_history(tmp_path):
    memory = ConversationMemoryAgent(tmp_path / "memory.sqlite3")
    memory.save_exchange("session-1", "What is RAG?", "RAG retrieves context before generation.")
    history = memory.load("session-1")
    assert [message["role"] for message in history] == ["user", "assistant"]
    assert "retrieves context" in history[1]["content"]
    assert memory.load("different-session") == []


def test_hit_at_k_and_mixed_answerable_dataset():
    assert hit_at_k(["wrong", "right"], ["right"], 3) == 1.0
    assert hit_at_k(["wrong"], ["right"], 1) == 0.0
    cases = [
        {"id": "q1", "question": "query one", "relevant_ids": ["doc-a"]},
        {"id": "q2", "question": "not present", "relevant_ids": []},
    ]
    report = evaluate_retrieval(cases, lambda question, k: ["doc-a"] if question == "query one" else ["doc-b"])
    assert report["query_count"] == 2
    assert report["answerable_query_count"] == 1
    assert report["unanswerable_query_count"] == 1
    assert report["metrics"]["top_1_accuracy"] == 1.0
    assert report["metrics_by_domain"]["Unspecified"]["answerable_query_count"] == 1


def test_evaluation_prefers_relevant_chunk_ids_over_document_ids():
    cases = [{"id": "q1", "domain": "HR", "question": "query", "relevant_ids": ["chunk-b"]}]
    ranked = [
        {"document_id": "doc-a", "chunk_id": "chunk-a", "rerank_score": 0.9},
        {"document_id": "doc-a", "chunk_id": "chunk-b", "rerank_score": 0.8},
    ]
    report = evaluate_retrieval(cases, lambda question, k: ranked[:k])
    assert report["metrics"]["top_1_accuracy"] == 0.0
    assert report["metrics"]["top_3_accuracy"] == 1.0
    assert report["metrics_by_domain"]["HR"]["metrics"]["top_3_accuracy"] == 1.0


def test_evaluation_labels_specific_relevant_chunks():
    chunks = [
        ChunkModel(chunk_id="chunk-a", document_id="doc-a", filename="policy.txt", file_type="txt", text="Employees receive 18 days of annual leave."),
        ChunkModel(chunk_id="chunk-b", document_id="doc-a", filename="policy.txt", file_type="txt", text="Sick leave requires manager notification."),
    ]
    assert find_relevant_chunk_ids(chunks, ["18 days of annual leave"]) == ["chunk-a"]
    assert find_relevant_chunk_ids(
        chunks,
        ["18 days of annual leave", "sick leave requires manager notification"],
    ) == ["chunk-a", "chunk-b"]
    assert find_relevant_chunk_ids(chunks, ["phrase not in the source"]) == []
