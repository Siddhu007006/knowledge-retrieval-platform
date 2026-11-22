"""Internal Pydantic models shared across the LangGraph workflow.

These are the typed contracts passed between graph nodes.  They are
distinct from the API request/response models in ``requests.py`` and
``responses.py``.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ChunkModel(BaseModel):
    """A single retrievable text chunk with provenance metadata."""

    chunk_id: str
    document_id: str
    filename: str
    file_type: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    text: str
    metadata: dict = Field(default_factory=dict)


class DocumentModel(BaseModel):
    """A parsed document before chunking."""

    document_id: str
    filename: str
    file_type: str
    source_text: str
    metadata: dict = Field(default_factory=dict)
    sections: List[dict] = Field(default_factory=list)


class RetrievedCandidate(BaseModel):
    """A candidate chunk returned by initial vector retrieval."""

    chunk: ChunkModel
    score: float


class RerankedChunk(BaseModel):
    """A chunk selected after the reranking stage."""

    chunk: ChunkModel
    score: float
    rerank_score: float


class ContextPackage(BaseModel):
    """The final filtered context handed to the generator."""

    question: str
    rewritten_query: str
    selected_chunks: List[RerankedChunk]
    source_metadata: List[dict] = Field(default_factory=list)


class GroundingResult(BaseModel):
    """The output of the groundedness checker."""

    status: str = Field(..., description="supported | partially_supported | unsupported")
    reasons: List[str] = Field(default_factory=list)
    supported_claims: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)


class WorkflowEvent(BaseModel):
    """A single workflow-stage event recorded during orchestration."""

    stage: str
    message: str
    latency_ms: Optional[float] = None
    data: dict = Field(default_factory=dict)
