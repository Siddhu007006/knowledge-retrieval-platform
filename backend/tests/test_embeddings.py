"""Unit tests for embeddings (mocked)."""

import numpy as np
import pytest

from app.retrieval.embeddings import embed_documents, embed_query, validate_dimension


class TestEmbeddings:
    def test_embed_query_returns_vector(self):
        vec = embed_query("hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert vec.shape[0] > 0

    def test_embed_documents_returns_matrix(self):
        vecs = embed_documents(["text one", "text two"])
        assert isinstance(vecs, np.ndarray)
        assert vecs.ndim == 2
        assert vecs.shape[0] == 2

    def test_embed_documents_empty(self):
        vecs = embed_documents([])
        assert vecs.size == 0

    def test_validate_dimension_match(self):
        vecs = np.zeros((3, 768), dtype=np.float32)
        validate_dimension(vecs, 768)

    def test_validate_dimension_mismatch(self):
        from app.core.exceptions import EmbeddingError

        vecs = np.zeros((3, 128), dtype=np.float32)
        with pytest.raises(EmbeddingError):
            validate_dimension(vecs, 768)
