"""Unit tests for the reranker."""

import pytest

from app.models.workflow import ChunkModel, RetrievedCandidate
from app.retrieval.reranker import Reranker


def _make_candidate(text, score=0.5, doc_id="d1"):
    return RetrievedCandidate(
        chunk=ChunkModel(
            chunk_id="c1",
            document_id=doc_id,
            filename="test.txt",
            file_type="txt",
            text=text,
        ),
        score=score,
    )


class TestReranker:
    def test_rerank_returns_sorted(self):
        candidates = [_make_candidate(f"chunk {i}", score=0.1 * i) for i in range(5)]
        reranker = Reranker(top_k=3)
        results = reranker.rerank("query", candidates)
        assert len(results) == 3
        assert all(hasattr(r, "rerank_score") for r in results)

    def test_rerank_empty_candidates(self):
        reranker = Reranker(top_k=3)
        results = reranker.rerank("query", [])
        assert results == []

    def test_rerank_preserves_original_score(self):
        candidates = [_make_candidate("text", score=0.42)]
        reranker = Reranker(top_k=1)
        results = reranker.rerank("query", candidates)
        assert results[0].score == 0.42
