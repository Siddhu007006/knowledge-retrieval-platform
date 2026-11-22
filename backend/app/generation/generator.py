"""Grounded answer generation using the Google Gen AI SDK.

The system instruction enforces:
  - use only the retrieved context
  - do not invent unsupported facts
  - distinguish evidence from inference
  - acknowledge missing information
  - provide source references
  - answer the user's original question clearly
"""

import json
import logging
from typing import Any, List, Optional


from app.core.config import settings
from app.core.exceptions import (
    GenerationError,
    MalformedModelResponseError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from app.core.retry import with_retry
from app.models.workflow import ContextPackage, RerankedChunk

logger = logging.getLogger("app.generator")

_CLIENT: Optional[Any] = None


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        from google import genai
        _CLIENT = genai.Client(api_key=settings.gemini_api_key)
    return _CLIENT


SYSTEM_INSTRUCTION = """You are a grounded retrieval-augmented generation assistant.

Rules:
1. Answer the user's ORIGINAL question clearly and concisely.
2. Use ONLY the provided retrieved context. Do not use outside knowledge.
3. Do not invent or fabricate facts that are not supported by the context.
4. If the context does not contain enough information to answer, say so explicitly.
5. Distinguish between facts directly stated in the evidence and reasonable inferences.
6. Reference sources using the bracketed source IDs provided (e.g. [S1], [S2]).
7. If multiple sources support a claim, cite all relevant source IDs.
8. Keep the answer focused and directly responsive to the question.
"""


def _build_context_text(chunks: List[RerankedChunk]) -> str:
    parts: List[str] = []
    for i, item in enumerate(chunks, 1):
        source_id = f"S{i}"
        meta_parts = [f"source_id={source_id}"]
        meta_parts.append(f"document={item.chunk.filename}")
        if item.chunk.page_number is not None:
            meta_parts.append(f"page={item.chunk.page_number}")
        if item.chunk.section:
            meta_parts.append(f"section={item.chunk.section}")
        meta_parts.append(f"chunk_id={item.chunk.chunk_id}")
        header = " | ".join(meta_parts)
        parts.append(f"[{header}]\n{item.chunk.text}")
    return "\n\n---\n\n".join(parts)


def _build_prompt(context: ContextPackage) -> str:
    context_text = _build_context_text(context.selected_chunks)
    return (
        f"User's original question: {context.question}\n\n"
        f"Retrieved evidence:\n{context_text}\n\n"
        f"Answer the user's original question using ONLY the evidence above. "
        f"Cite sources using [S1], [S2], etc. "
        f"If the evidence is insufficient, state that clearly."
    )


def generate_answer(context: ContextPackage) -> str:
    """Generate a grounded answer from the context package."""
    if not context.selected_chunks:
        return (
            "I don't have enough information from the provided documents to answer "
            "this question."
        )

    prompt = _build_prompt(context)
    client = _get_client()
    try:
        response = with_retry(lambda: client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config={
                "system_instruction":SYSTEM_INSTRUCTION,
                "temperature":0.2,
                "max_output_tokens":4096,
            },
        ))
        text = response.text
        if not text or not text.strip():
            raise MalformedModelResponseError("The model returned an empty response.")
        return text.strip()
    except (
        GenerationError,
        MalformedModelResponseError,
        ProviderError,
        ProviderRateLimitError,
        ProviderTimeoutError,
    ):
        raise
    except Exception as exc:
        logger.error(
            "Answer generation failed: %s: %s",
            type(exc).__name__,
            exc,
        )
        msg = str(exc).lower()
        if "timeout" in msg or "deadline" in msg:
            raise ProviderTimeoutError("Generation request timed out.") from exc
        if "rate" in msg and "limit" in msg:
            raise ProviderRateLimitError("Generation rate limit reached.") from exc
        raise GenerationError("Answer generation failed.") from exc
