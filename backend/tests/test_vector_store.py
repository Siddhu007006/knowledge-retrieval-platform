"""Unit tests for the vector store and FAISS index."""

import numpy as np
import pytest

from app.models.workflow import ChunkModel
from app.retrieval.embeddings import embed_documents, embed_query
from app.retrieval.vector_store import VectorStore
from app.storage.documents import DocumentStore
from app.storage.indexes import VectorIndex


class TestVectorIndex:
    def test_add_and_search(self):
        idx = VectorIndex()
        vectors = np.random.randn(5, 128).astype(np.float32)
        idx.add(vectors, ["c1", "c2", "c3", "c4", "c5"], ["d1", "d1", "d2", "d2", "d3"])
        assert idx.total() == 5

        query = np.random.randn(128).astype(np.float32)
        results = idx.search(query, k=3)
        assert len(results) <= 3
        assert all(isinstance(r[0], str) for r in results)

    def test_search_with_document_filter(self):
        idx = VectorIndex()
        vectors = np.random.randn(6, 64).astype(np.float32)
        chunk_ids = [f"c{i}" for i in range(6)]
        doc_ids = ["d1", "d1", "d2", "d2", "d3", "d3"]
        idx.add(vectors, chunk_ids, doc_ids)

        query = np.random.randn(64).astype(np.float32)
        results = idx.search(query, k=2, document_ids=["d1"])
        assert all(idx._meta[chunk_ids.index(r[0])]["document_id"] == "d1" for r in results) if results else True

    def test_remove_document(self):
        idx = VectorIndex()
        vectors = np.random.randn(4, 32).astype(np.float32)
        idx.add(vectors, ["c1", "c2", "c3", "c4"], ["d1", "d1", "d2", "d2"])
        removed = idx.remove_document("d1")
        assert removed == 2
        assert idx.total() == 2

    def test_empty_search(self):
        idx = VectorIndex()
        query = np.random.randn(32).astype(np.float32)
        results = idx.search(query, k=3)
        assert results == []


class TestVectorStore:
    def test_add_and_search_chunks(self):
        store = DocumentStore()
        idx = VectorIndex()
        vs = VectorStore(document_store=store, vector_index=idx)

        chunks = [
            ChunkModel(
                chunk_id=f"c{i}",
                document_id="d1",
                filename="test.txt",
                file_type="txt",
                text=f"chunk text {i}",
            )
            for i in range(3)
        ]
        for c in chunks:
            store._chunks[c.chunk_id] = c.model_dump()

        vs.add_chunks(chunks)
        assert vs.total_vectors() == 3

        results = vs.search("chunk text", k=2)
        assert len(results) <= 2
        for chunk, score in results:
            assert isinstance(chunk, ChunkModel)
