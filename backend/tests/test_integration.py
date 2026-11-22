"""Integration tests: upload → extraction → chunking → indexing → retrieval."""

import pytest

from app.ingestion.chunker import chunk_document
from app.ingestion.normalizer import normalize
from app.ingestion.base import get_parser
from app.models.workflow import ChunkModel
from app.retrieval.vector_store import VectorStore
from app.storage.documents import DocumentStore


class TestIngestionPipeline:
    def test_txt_upload_to_chunks(self):
        parser = get_parser("txt")
        doc = parser.parse(b"Hello world.\n\nSecond paragraph.", "test.txt", "d1")
        doc.source_text = normalize(doc.source_text)
        for s in doc.sections:
            s["text"] = normalize(s["text"])
        chunks = chunk_document(doc)
        assert len(chunks) >= 1
        assert all(c.document_id == "d1" for c in chunks)
        assert all(c.file_type == "txt" for c in chunks)

    def test_indexing_and_retrieval(self):
        store = DocumentStore()
        vs = VectorStore(document_store=store)

        chunks = [
            ChunkModel(
                chunk_id=f"c{i}",
                document_id="d1",
                filename="test.txt",
                file_type="txt",
                text=f"chunk text {i}",
            )
            for i in range(5)
        ]
        for c in chunks:
            store._chunks[c.chunk_id] = c.model_dump()

        vs.add_chunks(chunks)
        assert vs.total_vectors() == 5

        results = vs.search("chunk text 0", k=3)
        assert len(results) <= 3
        assert all(isinstance(c, ChunkModel) for c, _ in results)

    def test_document_deletion_removes_vectors(self):
        store = DocumentStore()
        vs = VectorStore(document_store=store)

        chunks_d1 = [
            ChunkModel(chunk_id="c1", document_id="d1", filename="a.txt", file_type="txt", text="text a"),
            ChunkModel(chunk_id="c2", document_id="d1", filename="a.txt", file_type="txt", text="text b"),
        ]
        chunks_d2 = [
            ChunkModel(chunk_id="c3", document_id="d2", filename="b.txt", file_type="txt", text="text c"),
        ]
        for c in chunks_d1 + chunks_d2:
            store._chunks[c.chunk_id] = c.model_dump()

        vs.add_chunks(chunks_d1)
        vs.add_chunks(chunks_d2)
        assert vs.total_vectors() == 3

        removed = vs.remove_document("d1")
        assert removed == 2
        assert vs.total_vectors() == 1

    def test_document_scope_filtering(self):
        store = DocumentStore()
        vs = VectorStore(document_store=store)

        chunks_d1 = [
            ChunkModel(chunk_id="c1", document_id="d1", filename="a.txt", file_type="txt", text="alpha"),
            ChunkModel(chunk_id="c2", document_id="d1", filename="a.txt", file_type="txt", text="beta"),
        ]
        chunks_d2 = [
            ChunkModel(chunk_id="c3", document_id="d2", filename="b.txt", file_type="txt", text="gamma"),
            ChunkModel(chunk_id="c4", document_id="d2", filename="b.txt", file_type="txt", text="delta"),
        ]
        for c in chunks_d1 + chunks_d2:
            store._chunks[c.chunk_id] = c.model_dump()

        vs.add_chunks(chunks_d1)
        vs.add_chunks(chunks_d2)

        results = vs.search("query", k=10, document_ids=["d1"])
        for chunk, _ in results:
            assert chunk.document_id == "d1"

    def test_deleted_document_not_retrievable(self):
        store = DocumentStore()
        vs = VectorStore(document_store=store)

        chunks = [
            ChunkModel(chunk_id="c1", document_id="d1", filename="a.txt", file_type="txt", text="text a"),
        ]
        for c in chunks:
            store._chunks[c.chunk_id] = c.model_dump()
        vs.add_chunks(chunks)

        vs.remove_document("d1")
        results = vs.search("text a", k=5)
        assert results == []
