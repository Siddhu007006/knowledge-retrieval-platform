"""Groundedness checker: validates whether the answer is supported by evidence.

The checker uses the Gemini model with a structured prompt that asks it to
classify the answer against the retrieved evidence and return a JSON object
with ``status``, ``supported_claims``, ``unsupported_claims``, and ``reasons``.

Status values:
  - ``supported``: all claims in the answer are backed by the evidence.
  - ``partially_supported``: some claims are backed; others are not.
  - ``unsupported``: the answer is not backed by the evidence.

This is a real validation step — it prevents unsupported generated content
from being presented as fully grounded.
"""

import json
import logging
import re
from typing import Any, Optional


from app.core.config import settings
from app.core.exceptions import GroundingError, MalformedModelResponseError
from app.core.retry import with_retry
from app.models.workflow import ContextPackage, GroundingResult

logger = logging.getLogger("app.grounding")

_CLIENT: Optional[Any] = None


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        from google import genai
        _CLIENT = genai.Client(api_key=settings.gemini_api_key)
    return _CLIENT


GROUNDING_INSTRUCTION = """You are a strict groundedness validator.

Given an answer and the retrieved evidence, determine whether the answer
is supported by the evidence. Return ONLY a JSON object with this exact shape:

{
  "status": "supported" | "partially_supported" | "unsupported",
  "supported_claims": ["claim 1", ...],
  "unsupported_claims": ["claim 1", ...],
  "reasons": ["reason 1", ...]
}

Rules:
- "supported" means every factual claim in the answer is directly stated in or
  reasonably inferable from the evidence.
- "partially_supported" means some claims are supported but others are not.
- "unsupported" means the answer contains facts not found in the evidence,
  or contradicts the evidence.
- Do not consider general knowledge; only the provided evidence matters.
- Return ONLY the JSON, no other text.
"""


def _build_grounding_prompt(context: ContextPackage, answer: str) -> str:
    evidence_parts = []
    for i, item in enumerate(context.selected_chunks, 1):
        evidence_parts.append(f"[S{i}] {item.chunk.text}")
    evidence = "\n\n".join(evidence_parts)
    return (
        f"Evidence:\n{evidence}\n\n"
        f"Answer to validate:\n{answer}\n\n"
        f"Evaluate the groundedness of the answer against the evidence above."
    )


def _parse_grounding_response(text: str) -> GroundingResult:
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find a JSON object in the text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError as exc:
                raise MalformedModelResponseError(
                    "Grounding response was not valid JSON."
                ) from exc
        else:
            raise MalformedModelResponseError(
                "Grounding response was not valid JSON."
            )

    status = data.get("status", "unsupported")
    if status not in ("supported", "partially_supported", "unsupported"):
        status = "unsupported"

    return GroundingResult(
        status=status,
        reasons=data.get("reasons", []),
        supported_claims=data.get("supported_claims", []),
        unsupported_claims=data.get("unsupported_claims", []),
    )


def check_grounding(context: ContextPackage, answer: str) -> GroundingResult:
    """Validate whether ``answer`` is supported by the evidence in ``context``."""
    if not context.selected_chunks:
        return GroundingResult(
            status="unsupported",
            reasons=["No evidence was provided."],
        )

    prompt = _build_grounding_prompt(context, answer)
    client = _get_client()
    try:
        response = with_retry(lambda: client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config={
                "system_instruction":GROUNDING_INSTRUCTION,
                "temperature":0.0,
                "max_output_tokens":2048,
                "response_mime_type":"application/json",
            },
        ))
        text = response.text
        if not text or not text.strip():
            raise MalformedModelResponseError("Grounding response was empty.")
        return _parse_grounding_response(text)
    except (GroundingError, MalformedModelResponseError):
        raise
    except Exception as exc:
        raise GroundingError("Groundedness validation failed.") from exc
