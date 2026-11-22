"""Workflow tests for the LangGraph query pipeline.

Tests the full multi-step workflow with mocked provider calls:
  1. Normal relevant query
  2. Document-scoped retrieval
  3. Multi-document retrieval
  4. Clarification branch and conversation memory
  5. Empty query
  6. Retry termination
  7. Invalid document scope
"""

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import fitz
import pytest
from docx import Document
from fastapi.testclient import TestClient

from app.main import app
from app.ingestion.chunker import chunk_document
from app.ingestion.normalizer import normalize
from app.ingestion.base import get_parser
from app.models.workflow import ChunkModel
from app.retrieval.vector_store import VectorStore
from app.storage.documents import DocumentStore


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def indexed_docs():
    """Pre-index two documents into the store for retrieval tests."""
    store = DocumentStore()
    vs = VectorStore(document_store=store)

    chunks_d1 = [
        ChunkModel(chunk_id="c1", document_id="d1", filename="doc1.txt", file_type="txt", text="The capital of France is Paris."),
        ChunkModel(chunk_id="c2", document_id="d1", filename="doc1.txt", file_type="txt", text="Python is a programming language."),
    ]
    chunks_d2 = [
        ChunkModel(chunk_id="c3", document_id="d2", filename="doc2.txt", file_type="txt", text="The Eiffel Tower is in Paris."),
        ChunkModel(chunk_id="c4", document_id="d2", filename="doc2.txt", file_type="txt", text="Machine learning uses neural networks."),
    ]
    all_chunks = chunks_d1 + chunks_d2
    for c in all_chunks:
        store._chunks[c.chunk_id] = c.model_dump()
    store._docs["d1"] = {
        "document_id": "d1", "filename": "doc1.txt", "file_type": "txt",
        "chunk_count": 2, "size_bytes": 100, "created_at": "2026-01-01T00:00:00Z",
    }
    store._docs["d2"] = {
        "document_id": "d2", "filename": "doc2.txt", "file_type": "txt",
        "chunk_count": 2, "size_bytes": 100, "created_at": "2026-01-01T00:00:00Z",
    }
    store._save()
    vs.add_chunks(all_chunks)
    return store, vs


class TestQueryWorkflow:
    def test_normal_query(self, indexed_docs):
        from app.graph.workflow import build_workflow
        wf = build_workflow()
        state = wf.run(question="What is the capital of France?")
        assert "answer" in state
        assert len(state["answer"]) > 0
        assert state.get("grounding_status") in ("supported", "partially_supported", "unsupported", "insufficient_context")

    def test_document_scoped_retrieval(self, indexed_docs):
        from app.graph.workflow import build_workflow
        wf = build_workflow()
        state = wf.run(question="What is the capital of France?", document_ids=["d1"])
        assert "answer" in state
        candidates = state.get("retrieved_candidates", [])
        for c in candidates:
            assert c.chunk.document_id == "d1"

    def test_multi_document_retrieval(self, indexed_docs):
        from app.graph.workflow import build_workflow
        wf = build_workflow()
        state = wf.run(question="What is in Paris?", document_ids=["d1", "d2"])
        assert "answer" in state
        candidates = state.get("retrieved_candidates", [])
        doc_ids = {c.chunk.document_id for c in candidates}
        assert doc_ids.issubset({"d1", "d2"})

    def test_empty_query_raises(self):
        from app.core.exceptions import EmptyQueryError
        from app.graph.workflow import build_workflow
        wf = build_workflow()
        with pytest.raises(EmptyQueryError):
            wf.run(question="")

    def test_unrelated_documents_not_retrieved(self, indexed_docs):
        from app.graph.workflow import build_workflow
        wf = build_workflow()
        state = wf.run(question="What is the capital of France?", document_ids=["d2"])
        candidates = state.get("retrieved_candidates", [])
        for c in candidates:
            assert c.chunk.document_id == "d2"

    def test_retry_loop_terminates(self, indexed_docs, monkeypatch):
        """Ensure the workflow terminates even when grounding always fails."""
        monkeypatch.setattr("app.core.config.settings.grounding_max_retries", 1)

        # Make grounding always return unsupported
        with patch("app.generation.grounding._get_client") as gc:
            client = MagicMock()
            client.models.generate_content.return_value = MagicMock(
                text=json.dumps({
                    "status": "unsupported",
                    "supported_claims": [],
                    "unsupported_claims": ["claim"],
                    "reasons": ["No evidence."],
                })
            )
            gc.return_value = client

            from app.graph.workflow import build_workflow
            wf = build_workflow()
            state = wf.run(question="What is the capital of France?")
            assert state.get("retry_count", 0) <= 1
            assert state.get("grounding_status") in ("unsupported", "insufficient_context")


class TestAPIEndpoints:
    def test_upload_txt(self, client):
        resp = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", b"Hello world. This is a test.", "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "document_id" in data
        assert data["filename"] == "test.txt"
        assert data["chunk_count"] >= 1

    def test_upload_csv(self, client):
        resp = client.post(
            "/api/v1/documents/upload",
            files={"file": ("employees.csv", b"name,team\nAsha,Engineering", "text/csv")},
        )
        assert resp.status_code == 200
        assert resp.json()["file_type"] == "csv"

    def test_upload_pdf_and_docx_are_extracted_and_indexed(self, client):
        pdf_document = fitz.open()
        pdf_page = pdf_document.new_page()
        pdf_page.insert_text((72, 72), "PDF knowledge base indexing validation.")
        pdf_content = pdf_document.tobytes()
        pdf_document.close()

        word_document = Document()
        word_document.add_heading("DOCX knowledge base", level=1)
        word_document.add_paragraph("DOCX extraction and indexing validation.")
        word_buffer = BytesIO()
        word_document.save(word_buffer)

        uploads = [
            ("sample.pdf", pdf_content, "application/pdf", "pdf"),
            ("sample.docx", word_buffer.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"),
        ]
        for filename, content, mime_type, expected_type in uploads:
            response = client.post(
                "/api/v1/documents/upload",
                files={"file": (filename, content, mime_type)},
            )
            assert response.status_code == 200
            payload = response.json()
            assert payload["file_type"] == expected_type
            assert payload["chunk_count"] >= 1

    def test_upload_unsupported_markdown(self, client):
        resp = client.post(
            "/api/v1/documents/upload",
            files={"file": ("notes.md", b"# outside milestone", "text/markdown")},
        )
        assert resp.status_code == 415

    def test_upload_unsupported_type(self, client):
        resp = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.xyz", b"data", "application/octet-stream")},
        )
        assert resp.status_code == 415

    def test_upload_empty_file(self, client):
        resp = client.post(
            "/api/v1/documents/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert resp.status_code == 400

    def test_list_documents(self, client):
        client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", b"Hello world.", "text/plain")},
        )
        resp = client.get("/api/v1/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    def test_delete_document(self, client):
        upload = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", b"Hello world.", "text/plain")},
        )
        doc_id = upload.json()["document_id"]
        resp = client.delete(f"/api/v1/documents/{doc_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    def test_delete_nonexistent(self, client):
        resp = client.delete("/api/v1/documents/nonexistent-id")
        assert resp.status_code == 404

    def test_empty_query(self, client):
        resp = client.post("/api/v1/query", json={"question": ""})
        assert resp.status_code == 400

    def test_query_too_long(self, client, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.max_query_length", 10)
        resp = client.post("/api/v1/query", json={"question": "This is a very long query that exceeds the limit"})
        assert resp.status_code == 400

    def test_both_document_id_and_ids_rejected(self, client):
        resp = client.post("/api/v1/query", json={
            "question": "test",
            "document_id": "d1",
            "document_ids": ["d1"],
        })
        assert resp.status_code == 400

    def test_query_response_schema(self, client):
        client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", b"The capital of France is Paris.", "text/plain")},
        )
        resp = client.post("/api/v1/query", json={"question": "What is the capital of France?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "conversation_id" in data
        assert "clarification_required" in data
        assert "citations" in data
        assert "grounding_status" in data
        assert "retrieval_metadata" in data
        assert "candidate_count" in data["retrieval_metadata"]
        assert "reranked_count" in data["retrieval_metadata"]
        assert "selected_count" in data["retrieval_metadata"]



class TestConversationAndClarification:
    def test_unresolved_reference_returns_clarification(self, client):
        response = client.post("/api/v1/query", json={"question": "What about it?"})
        assert response.status_code == 200
        assert response.json()["clarification_required"] is True
        assert "refer" in response.json()["answer"].lower()

    def test_follow_up_uses_session_history(self, client, indexed_docs):
        first = client.post("/api/v1/query", json={"question": "What is the capital of France?"})
        conversation_id = first.json()["conversation_id"]
        follow_up = client.post("/api/v1/query", json={"question": "What about it?", "conversation_id": conversation_id})
        assert follow_up.status_code == 200
        assert follow_up.json()["clarification_required"] is False


    def test_partially_supported_answer_abstains_after_retries(self, monkeypatch):
        """Partially supported answers must retry and eventually abstain."""
        from app.graph.edges import route_grounding
        from app.graph.nodes import handle_unsupported_grounding
        from app.models.workflow import GroundingResult

        monkeypatch.setattr(
            "app.core.config.settings.grounding_max_retries", 1
        )

        state = {
            "grounding_status": "partially_supported",
            "retry_count": 0,
            "request_id": "test-request",
            "events": [],
        }

        # First attempt: retry.
        assert route_grounding(state) == "rewrite_query"

        # After the retry is exhausted: safe fallback.
        state["retry_count"] = 1

        assert route_grounding(state) == "handle_unsupported_grounding"

        state["answer"] = "This answer contains an unsupported claim."
        state["grounding_result"] = GroundingResult(
            status="partially_supported",
            supported_claims=["One supported claim"],
            unsupported_claims=["One unsupported claim"],
            reasons=["Part of the answer lacks evidence."],
        )

        result = handle_unsupported_grounding(state)

        assert result["grounding_status"] == "insufficient_context"
        assert result["answer"] != state["answer"]
        assert "couldn't verify" in result["answer"]
        assert result["citations"] == []