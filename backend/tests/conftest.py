"""Shared test fixtures and provider mocks.

All automated tests mock the Gemini API calls so they run without a real
API key.  Live retrieval evaluation is run separately using the labelled sample dataset.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

FAKE_DIM = 768


class _FakeEmbedding:
    def __init__(self, values):
        self.values = values


class _FakeEmbedResult:
    def __init__(self, embeddings):
        self.embeddings = embeddings


def _make_fake_embed_result(texts):
    """Return a fake ``embed_content`` result with ``.embeddings`` list."""
    if isinstance(texts, str):
        texts = [texts]
    embeddings = []
    for t in texts:
        rng = np.random.RandomState(hash(str(t)) % (2**32 - 1))
        embeddings.append(_FakeEmbedding(rng.randn(FAKE_DIM).astype(np.float32).tolist()))
    return _FakeEmbedResult(embeddings)


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path, monkeypatch):
    """Redirect all storage to a temp directory so tests don't touch real data."""
    monkeypatch.setattr("app.core.config.settings.storage_dir", str(tmp_path / "storage"))
    monkeypatch.setattr("app.core.config.settings.index_dir", str(tmp_path / "index"))
    monkeypatch.setattr("app.core.config.settings.conversation_db_path", str(tmp_path / "storage" / "conversations.sqlite3"))
    yield

@pytest.fixture(autouse=True)
def _mock_gemini(tmp_path, monkeypatch):
    """Mock Gemini and isolate rate-limit accounting during tests."""

    from app.retrieval.rate_limiter import EmbeddingRateLimiter

    test_limiter = EmbeddingRateLimiter(
        rpm_limit=1_000_000,
        tpm_limit=1_000_000_000,
        rpd_limit=1_000_000_000,
        state_path=tmp_path / "test_embedding_usage.json",
    )

    # Tests must never consume the production quota tracker.
    monkeypatch.setattr(
        "app.retrieval.embeddings._RATE_LIMITER",
        test_limiter,
    )

    with patch("app.retrieval.embeddings._get_client") as c1, \
         patch("app.generation.generator._get_client") as c2, \
         patch("app.generation.grounding._get_client") as c3, \
         patch("app.generation.query_rewriter._get_client") as c4:

        for mock_client in (c1, c2, c3, c4):
            client = MagicMock()
            client.models.embed_content.side_effect = (
                lambda model, contents, config=None:
                    _make_fake_embed_result(contents)
            )
            client.models.generate_content.return_value = MagicMock(
                text="This is a grounded answer. [S1]"
            )
            mock_client.return_value = client

        c3.return_value.models.generate_content.return_value = MagicMock(
            text=json.dumps({
                "status": "supported",
                "supported_claims": ["claim 1"],
                "unsupported_claims": [],
                "reasons": ["All claims are backed by evidence."],
            })
        )

        c4.return_value.models.generate_content.return_value = MagicMock(
            text=json.dumps({
                "rewritten_query": "rewritten query",
                "reason": "Improved for retrieval.",
            })
        )

        yield

@pytest.fixture
def fake_embedding_dim():
    return FAKE_DIM