
"""Centralized Gemini embedding generation with quota-aware batching."""

import logging
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np

from app.core.config import settings
from app.core.exceptions import (
    EmbeddingError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from app.core.retry import with_retry
from app.retrieval.rate_limiter import (
    EmbeddingRateLimiter,
    EmbeddingQuotaBudgetExceeded,
)

logger = logging.getLogger("app.embeddings")

_CLIENT: Optional[Any] = None

_RATE_LIMITER = EmbeddingRateLimiter(
    rpm_limit=settings.embedding_rpm_limit,
    tpm_limit=settings.embedding_tpm_limit,
    rpd_limit=settings.embedding_rpd_limit,
    state_path=settings.storage_path() / "embedding_usage.json",
    chars_per_token=settings.embedding_chars_per_token,
)


def _get_client():
    global _CLIENT

    if _CLIENT is None:
        from google import genai

        _CLIENT = genai.Client(
            api_key=settings.gemini_api_key
        )

    return _CLIENT


def _iter_batches(
    texts: List[str],
) -> Iterator[Tuple[List[str], int]]:
    """Split inputs by both maximum batch size and estimated TPM."""

    batch_size = min(
        settings.embedding_batch_size,
        100,  # Gemini synchronous embedding batch maximum.
        settings.embedding_rpm_limit,
    )

    if batch_size <= 0:
        raise EmbeddingError("Invalid embedding batch size.")

    current_batch: List[str] = []
    current_tokens = 0

    for text in texts:
        token_count = _RATE_LIMITER.estimate_tokens(text)

        if token_count > settings.embedding_tpm_limit:
            raise EmbeddingQuotaBudgetExceeded(
                "A single chunk exceeds the configured TPM budget. "
                "Review the chunking configuration."
            )

        exceeds_count = len(current_batch) >= batch_size
        exceeds_tokens = (
            current_batch
            and current_tokens + token_count
            > settings.embedding_tpm_limit
        )

        if exceeds_count or exceeds_tokens:
            yield current_batch, current_tokens
            current_batch = []
            current_tokens = 0

        current_batch.append(text)
        current_tokens += token_count

    if current_batch:
        yield current_batch, current_tokens


def _embed(texts: List[str], task_type: str) -> List[List[float]]:
    """Generate ordered embeddings with RPM, TPM and RPD protection."""

    if not texts:
        return []

    client = _get_client()
    all_vectors: List[List[float]] = []

    try:
        for batch_number, (batch, batch_tokens) in enumerate(
            _iter_batches(texts),
            start=1,
        ):

            def _call() -> list:
                # Reserve quota for every input before each API attempt.
                # This also runs again when with_retry retries the request.
                _RATE_LIMITER.wait_for_capacity(
                    input_count=len(batch),
                    estimated_tokens=batch_tokens,
                )

                result = client.models.embed_content(
                    model=settings.gemini_embedding_model,
                    contents=batch,
                    config={"task_type": task_type},
                )

                embeddings = result.embeddings or []

                if len(embeddings) != len(batch):
                    raise EmbeddingError(
                        f"Expected {len(batch)} embeddings, "
                        f"received {len(embeddings)}."
                    )

                vectors = [item.values for item in embeddings]

                if any(not vector for vector in vectors):
                    raise EmbeddingError(
                        "Gemini returned an empty embedding vector."
                    )

                return vectors

            batch_vectors = with_retry(_call)
            all_vectors.extend(batch_vectors)

            logger.info(
                "Embedding batch completed: batch=%d, "
                "batch_size=%d, batch_tokens=%d, "
                "total_processed=%d/%d",
                batch_number,
                len(batch),
                batch_tokens,
                len(all_vectors),
                len(texts),
            )

        if len(all_vectors) != len(texts):
            raise EmbeddingError(
                f"Expected {len(texts)} vectors, "
                f"received {len(all_vectors)}."
            )

        dimensions = {
            len(vector) for vector in all_vectors
        }

        if len(dimensions) != 1:
            raise EmbeddingError(
                "Embedding batches returned inconsistent dimensions."
            )

        return all_vectors

    except Exception as exc:
        logger.exception(
            "Embedding generation failed: model=%s, "
            "task_type=%s, input_count=%d",
            settings.gemini_embedding_model,
            task_type,
            len(texts),
        )

        if isinstance(exc, EmbeddingError):
            raise

        message = str(exc).lower()

        if "timeout" in message or "deadline" in message:
            raise ProviderTimeoutError(
                "Embedding request timed out."
            ) from exc

        if "429" in message or "rate limit" in message:
            raise ProviderRateLimitError(
                "Embedding rate limit reached."
            ) from exc

        if isinstance(exc, EmbeddingQuotaBudgetExceeded):
            raise EmbeddingError(str(exc)) from exc

        raise EmbeddingError(
            "Failed to generate embeddings."
        ) from exc


def embed_documents(texts: List[str]) -> np.ndarray:
    """Embed document chunks using RETRIEVAL_DOCUMENT."""
    vectors = _embed(texts, "RETRIEVAL_DOCUMENT")
    return np.asarray(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a user question using RETRIEVAL_QUERY."""
    vectors = _embed([text], "RETRIEVAL_QUERY")
    return np.asarray(vectors[0], dtype=np.float32)


def embedding_dimension() -> int:
    """Probe the model's embedding dimension."""
    vector = embed_query("dimensionality probe")
    return int(vector.shape[0])


def validate_dimension(
    vectors: np.ndarray,
    expected: int,
) -> None:
    if vectors.size == 0:
        return

    actual = vectors.shape[-1]

    if actual != expected:
        raise EmbeddingError(
            f"Embedding dimension mismatch: "
            f"expected {expected}, got {actual}."
        )
