"""Pydantic response models for every API endpoint."""

from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentMetadataResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    chunk_count: int
    size_bytes: int
    created_at: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentMetadataResponse]
    total: int


class DocumentDeleteResponse(BaseModel):
    document_id: str
    deleted: bool = True


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    chunk_count: int
    size_bytes: int


class Citation(BaseModel):
    document_id: str
    filename: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    chunk_id: str
    excerpt: str = Field(..., description="A short excerpt from the source chunk.")


class RetrievalMetadata(BaseModel):
    candidate_count: int = 0
    reranked_count: int = 0
    selected_count: int = 0
    rewrite_applied: bool = False
    rewritten_query: Optional[str] = None
    rewrite_reason: Optional[str] = None
    retry_count: int = 0


class QueryResponse(BaseModel):
    conversation_id: Optional[str] = None
    clarification_required: bool = False
    answer: str
    citations: List[Citation]
    grounding_status: str = Field(
        ..., description="supported | partially_supported | unsupported | insufficient_context | clarification_required"
    )
    retrieval_metadata: RetrievalMetadata


class ErrorResponse(BaseModel):
    error: dict

    class Config:
        arbitrary_types_allowed = True
