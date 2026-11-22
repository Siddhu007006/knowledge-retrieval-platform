"""Configurable, section-aware chunking strategy.

Strategy
--------
The chunker walks the document's ``sections`` (as produced by the format
parser).  Each section is split into paragraphs.  Paragraphs are greedily
accumulated into chunks of approximately ``CHUNK_SIZE`` tokens, with
``CHUNK_OVERLAP`` tokens of overlap between consecutive chunks.

Token estimation uses a lightweight heuristic (``~4 chars per token``)
which avoids a hard tokenizer dependency while keeping chunk sizes in the
target 600-800 token range.

For PDF, page numbers are carried through. For DOCX, paragraph styles and
table locations are retained. For CSV, column-labelled rows remain distinct.
For TXT, paragraph boundaries are respected.
"""

import uuid
from typing import List

from app.core.config import settings
from app.models.workflow import ChunkModel, DocumentModel

_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _split_long_paragraph(text: str, max_tokens: int) -> List[str]:
    """Split a single paragraph that exceeds ``max_tokens`` into sentences."""
    if _estimate_tokens(text) <= max_tokens:
        return [text]
    sentences = text.split(". ")
    pieces: List[str] = []
    current = ""
    for sent in sentences:
        candidate = (current + ". " + sent) if current else sent
        if _estimate_tokens(candidate) > max_tokens and current:
            pieces.append(current.strip())
            current = sent
        else:
            current = candidate
    if current:
        pieces.append(current.strip())
    return pieces


def chunk_document(document: DocumentModel) -> List[ChunkModel]:
    """Produce a list of ``ChunkModel`` objects from a parsed document."""
    target_tokens = settings.chunk_size
    overlap_tokens = settings.chunk_overlap
    max_chars = target_tokens * _CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * _CHARS_PER_TOKEN

    chunks: List[ChunkModel] = []
    prev_tail = ""

    for section in document.sections:
        section_text = section["text"].strip()
        if not section_text:
            continue
        page_number = section.get("page_number")
        section_name = section.get("section")

        paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]
        buf = "" if document.file_type == "csv" else prev_tail

        for para in paragraphs:
            sub_paras = _split_long_paragraph(para, target_tokens)
            for sub in sub_paras:
                candidate = (buf + "\n\n" + sub) if buf else sub
                if _estimate_tokens(candidate) > target_tokens and buf:
                    chunk = _make_chunk(document, buf, page_number, section_name)
                    chunks.append(chunk)
                    prev_tail = buf[-overlap_chars:] if overlap_chars > 0 else ""
                    buf = (prev_tail + "\n\n" + sub) if prev_tail else sub
                else:
                    buf = candidate
        if buf:
            chunk = _make_chunk(document, buf, page_number, section_name)
            chunks.append(chunk)
            prev_tail = buf[-overlap_chars:] if overlap_chars > 0 else ""
            buf = ""

    if buf:
        chunks.append(_make_chunk(document, buf, None, None))

    return chunks


def _make_chunk(
    document: DocumentModel,
    text: str,
    page_number: int | None,
    section_name: str | None,
) -> ChunkModel:
    return ChunkModel(
        chunk_id=str(uuid.uuid4()),
        document_id=document.document_id,
        filename=document.filename,
        file_type=document.file_type,
        page_number=page_number,
        section=section_name,
        text=text.strip(),
        metadata={
            "source_text_length": len(text),
            "estimated_tokens": _estimate_tokens(text),
        },
    )
