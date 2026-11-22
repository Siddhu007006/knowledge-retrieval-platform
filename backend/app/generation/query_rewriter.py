"""Query rewriting / refinement for better retrieval.

Uses the Gemini model to rewrite ambiguous questions, resolve pronouns,
and improve conversational follow-ups.  The original question is always
preserved — the rewrite is only used for retrieval.

If rewriting fails for any reason, the caller falls back to the original
query (handled in the graph node).
"""

import json
import logging
import re
from typing import Any, Optional

from pydantic import BaseModel

from app.core.config import settings
from app.core.exceptions import GenerationError, MalformedModelResponseError
from app.core.retry import with_retry

logger = logging.getLogger("app.query_rewriter")

_CLIENT: Optional[Any] = None


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        from google import genai
        _CLIENT = genai.Client(api_key=settings.gemini_api_key)
    return _CLIENT


REWRITE_INSTRUCTION = """You are a query rewriting assistant for a retrieval system.

Given a user's question, rewrite it to improve retrieval quality. Do NOT
answer the question — only rewrite it for better search.

Rewrite when:
- The question contains pronouns or vague references (e.g., "What about that one?")
- The question is a conversational follow-up (e.g., "And how does it work?")
- The question is ambiguous or too short for good retrieval
- The question would benefit from more specific terms

Do NOT rewrite when:
- The question is already clear and specific
- The question is a straightforward factual query
- Rewriting would change the meaning

Return ONLY a JSON object with this exact shape:
{
  "rewritten_query": "the rewritten or original query",
  "reason": "brief explanation of what was changed or why no change was made"
}
"""


class RewriteResult(BaseModel):
    rewritten_query: str
    reason: str


def _parse_rewrite_response(text: str) -> RewriteResult:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError as exc:
                raise MalformedModelResponseError(
                    "Query rewrite response was not valid JSON."
                ) from exc
        else:
            raise MalformedModelResponseError(
                "Query rewrite response was not valid JSON."
            )
    return RewriteResult(
        rewritten_query=data.get("rewritten_query", ""),
        reason=data.get("reason", "No reason provided."),
    )


def rewrite(
    question: str,
    document_ids: Optional[list[str]] = None,
    retry_count: int = 0,
    conversation_history: Optional[list[dict[str, str]]] = None,
) -> RewriteResult:
    """Rewrite a question for better retrieval.

    On retry (retry_count > 0), the prompt asks for an alternative
    formulation to try a different retrieval path.
    """
    prompt = f"User question: {question}"
    if conversation_history:
        history_text = "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in conversation_history[-8:])
        prompt += f"\n\nRecent conversation (use only to resolve references in the current question):\n{history_text}"
    if retry_count > 0:
        prompt += (
            f"\n\nNote: This is retry #{retry_count}. The previous retrieval "
            f"attempt did not find sufficient results. Please provide an "
            f"alternative formulation with different terminology or phrasing."
        )
    if document_ids:
        prompt += (
            f"\n\nThe search will be scoped to {len(document_ids)} specific "
            f"document(s)."
        )

    client = _get_client()
    try:
        response = with_retry(lambda: client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config={
                "system_instruction":REWRITE_INSTRUCTION,
                "temperature":0.3,
                "max_output_tokens":2048,
                "response_mime_type":"application/json",
            },
        ))
        text = response.text
        if not text or not text.strip():
            raise MalformedModelResponseError("Query rewrite response was empty.")
        result = _parse_rewrite_response(text)
        if not result.rewritten_query:
            result.rewritten_query = question
        return result
    except (GenerationError, MalformedModelResponseError):
        raise
    except Exception as exc:
        raise GenerationError("Query rewriting failed.") from exc
