"""Pydantic request models for every API endpoint."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class QueryRequest(BaseModel):
    question: str = Field(..., description="The user's question.")
    conversation_id: Optional[str] = Field(default=None, description="Session identifier used to load and persist conversation history.")
    document_id: Optional[str] = Field(
        default=None, description="Scope retrieval to a single document."
    )
    document_ids: Optional[List[str]] = Field(
        default=None, description="Scope retrieval to multiple documents."
    )

    @field_validator("question")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            from app.core.exceptions import EmptyQueryError

            raise EmptyQueryError("The query must not be empty.")
        return stripped

    @model_validator(mode="after")
    def _check_scope(self) -> "QueryRequest":
        if self.document_id is not None and self.document_ids is not None:
            from app.core.exceptions import InvalidDocumentScopeError

            raise InvalidDocumentScopeError(
                "Specify either document_id or document_ids, not both."
            )
        if self.document_ids is not None:
            if len(self.document_ids) == 0:
                from app.core.exceptions import InvalidDocumentScopeError

                raise InvalidDocumentScopeError("document_ids must not be empty.")
            if len(set(self.document_ids)) != len(self.document_ids):
                from app.core.exceptions import InvalidDocumentScopeError

                raise InvalidDocumentScopeError("document_ids must not contain duplicates.")
        return self
