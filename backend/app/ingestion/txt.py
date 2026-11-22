"""Plain-text parser.

Preserves paragraph boundaries (double-newline separated blocks).
"""

from app.models.workflow import DocumentModel
from app.ingestion.base import DocumentParser, register_parser


@register_parser("txt")
class TXTParser(DocumentParser):
    def parse(self, content: bytes, filename: str, document_id: str) -> DocumentModel:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1", errors="replace")

        self._empty_check(text, filename)

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        sections = [{"page_number": None, "text": p, "section": None} for p in paragraphs]

        return DocumentModel(
            document_id=document_id,
            filename=filename,
            file_type="txt",
            source_text=text,
            metadata={"paragraph_count": len(paragraphs)},
            sections=sections,
        )
