"""Document upload, list, and delete endpoints."""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, File, UploadFile

from app.core.config import settings
from app.core.exceptions import (
    DocumentNotFoundError,
    EmptyFileError,
    FileTooLargeError,
    InvalidFileError,
    UnsupportedFileTypeError,
)
from app.core.logging import log_event
from app.ingestion import get_parser, supported_types
from app.ingestion.chunker import chunk_document
from app.ingestion.normalizer import normalize
from app.models.responses import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentMetadataResponse,
    DocumentUploadResponse,
)
from app.retrieval.vector_store import VectorStore
from app.storage.documents import DocumentStore

router = APIRouter()

_ALLOWED_EXTENSIONS = set(supported_types())


def _sanitize_filename(filename: str) -> str:
    """Remove path components and dangerous characters from filename."""
    import os

    safe = os.path.basename(filename)
    safe = safe.replace("..", "").replace("/", "").replace("\\", "")
    return safe


def _get_extension(filename: str) -> str:
    import os

    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    return ext


def _get_store_and_vector_store() -> tuple[DocumentStore, VectorStore]:
    store = DocumentStore()
    vs = VectorStore(document_store=store)
    return store, vs


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    filename = _sanitize_filename(file.filename or "unknown")
    ext = _get_extension(filename)

    if ext not in _ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"File type '.{ext}' is not supported. Supported types: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
        )

    content = await file.read()
    if not content:
        raise EmptyFileError(f"The file '{filename}' is empty.")

    if len(content) > settings.max_file_size_bytes:
        raise FileTooLargeError(
            f"File size {len(content)} bytes exceeds maximum {settings.max_file_size_bytes} bytes."
        )

    document_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    log_event(
        "Document upload started",
        request_id=request_id,
        document_id=document_id,
        filename=filename,
        size_bytes=len(content),
    )

    parser = get_parser(ext)
    document = parser.parse(content=content, filename=filename, document_id=document_id)
    document.source_text = normalize(document.source_text)
    for section in document.sections:
        section["text"] = normalize(section["text"])

    chunks = chunk_document(document)
    if not chunks:
        raise InvalidFileError(f"No chunks could be extracted from '{filename}'.")

    store, vs = _get_store_and_vector_store()
    created_at = datetime.now(timezone.utc).isoformat()
    # IMPORTANT: Add to vector store FIRST. If embedding fails, document metadata is not saved.
    # This prevents incomplete records if embedding generation fails.
    vs.add_chunks(chunks)
    store.save_document(document, chunks, len(content), created_at)

    log_event(
        "Document uploaded",
        request_id=request_id,
        document_id=document_id,
        filename=filename,
        chunk_count=len(chunks),
    )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=filename,
        file_type=ext,
        chunk_count=len(chunks),
        size_bytes=len(content),
    )


@router.get("/documents", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    store = DocumentStore()
    docs = store.list_documents()
    return DocumentListResponse(
        documents=[DocumentMetadataResponse(**d) for d in docs],
        total=len(docs),
    )


@router.delete("/documents/{document_id}", response_model=DocumentDeleteResponse)
def delete_document(document_id: str) -> DocumentDeleteResponse:
    store, vs = _get_store_and_vector_store()
    if not store.document_exists(document_id):
        raise DocumentNotFoundError(f"Document '{document_id}' was not found.")

    store.delete_document(document_id)
    vs.remove_document(document_id)

    log_event(
        "Document deleted",
        document_id=document_id,
    )
    return DocumentDeleteResponse(document_id=document_id, deleted=True)
