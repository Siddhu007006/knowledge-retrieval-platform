"""Application-wide exception hierarchy.

Every error surfaced through the API is wrapped in ``AppError`` so the
response shape is uniform.  Internal details (stack traces, file paths,
provider internals) are never included in ``message`` or ``details``.
"""

from typing import Any, List, Optional


class AppError(Exception):
    """Base application error with a stable code/message/details shape."""

    code: str = "internal_error"
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        code: Optional[str] = None,
        details: Optional[List[Any]] = None,
    ) -> None:
        if message:
            self.message = message
        if code:
            self.code = code
        self.details = details or []
        super().__init__(self.message)

    def to_dict(self) -> dict:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


class InvalidFileError(AppError):
    code = "invalid_file"
    message = "The uploaded file is invalid or unreadable."


class UnsupportedFileTypeError(AppError):
    code = "unsupported_file_type"
    message = "The uploaded file type is not supported."


class EmptyFileError(AppError):
    code = "empty_file"
    message = "The uploaded file is empty."


class FileTooLargeError(AppError):
    code = "file_too_large"
    message = "The uploaded file exceeds the maximum allowed size."


class MalformedDocumentError(AppError):
    code = "malformed_document"
    message = "The document could not be parsed."


class EmptyQueryError(AppError):
    code = "empty_query"
    message = "The query must not be empty."


class QueryTooLongError(AppError):
    code = "query_too_long"
    message = "The query exceeds the maximum allowed length."


class InvalidDocumentScopeError(AppError):
    code = "invalid_document_scope"
    message = "The document scope is invalid."


class DocumentNotFoundError(AppError):
    code = "document_not_found"
    message = "The requested document was not found."


class EmbeddingError(AppError):
    code = "embedding_failure"
    message = "Failed to generate embeddings."


class VectorSearchError(AppError):
    code = "vector_search_failure"
    message = "Vector search failed."


class RerankerError(AppError):
    code = "reranker_failure"
    message = "Reranking failed."


class GenerationError(AppError):
    code = "generation_failure"
    message = "Answer generation failed."


class GroundingError(AppError):
    code = "grounding_failure"
    message = "Groundedness validation failed."


class ProviderTimeoutError(AppError):
    code = "provider_timeout"
    message = "The AI provider took too long to respond."


class ProviderRateLimitError(AppError):
    code = "provider_rate_limit"
    message = "The AI provider rate limit was reached."


class ProviderError(AppError):
    code = "provider_error"
    message = "The AI provider returned an error."


class MalformedModelResponseError(AppError):
    code = "malformed_model_response"
    message = "The model response could not be parsed."



class InsufficientContextError(AppError):
    code = "insufficient_context"
    message = "Insufficient context was found to answer the question."


class QueryValidationError(AppError):
    code = "query_validation_error"
    message = "The query failed validation."
