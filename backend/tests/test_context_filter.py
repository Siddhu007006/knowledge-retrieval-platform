"""Unit tests for the context filter."""

from app.models.workflow import ChunkModel, RerankedChunk
from app.retrieval.context_filter import ContextFilter


def _make_reranked(text, score=0.5, doc_id="d1"):
    return RerankedChunk(
        chunk=ChunkModel(
            chunk_id="c1",
            document_id=doc_id,
            filename="test.txt",
            file_type="txt",
            text=text,
        ),
        score=score,
        rerank_score=score,
    )


class TestContextFilter:
    def test_filter_removes_duplicates(self):
        cf = ContextFilter(min_score=0.0)
        chunks = [_make_reranked("same text", 0.8), _make_reranked("same text", 0.7)]
        result = cf.filter("q", "q", chunks)
        assert len(result.selected_chunks) == 1

    def test_filter_removes_weak_candidates(self):
        cf = ContextFilter(min_score=0.5)
        chunks = [_make_reranked("good", 0.8), _make_reranked("weak", 0.1)]
        result = cf.filter("q", "q", chunks)
        assert len(result.selected_chunks) == 1
        assert result.selected_chunks[0].chunk.text == "good"

    def test_filter_preserves_scope(self):
        cf = ContextFilter(min_score=0.0)
        chunks = [
            _make_reranked("text a", 0.8, doc_id="d1"),
            _make_reranked("text b", 0.7, doc_id="d2"),
        ]
        result = cf.filter("q", "q", chunks, document_ids=["d1"])
        assert len(result.selected_chunks) == 1
        assert result.selected_chunks[0].chunk.document_id == "d1"

    def test_is_sufficient(self):
        cf = ContextFilter(min_score=0.0)
        result = cf.filter("q", "q", [_make_reranked("text", 0.8)])
        assert cf.is_sufficient(result)

    def test_is_not_sufficient(self):
        cf = ContextFilter(min_score=0.0)
        result = cf.filter("q", "q", [])
        assert not cf.is_sufficient(result)

    def test_source_metadata(self):
        cf = ContextFilter(min_score=0.0)
        chunks = [_make_reranked("text", 0.8)]
        result = cf.filter("q", "q", chunks)
        assert len(result.source_metadata) == 1
        assert result.source_metadata[0]["document_id"] == "d1"
