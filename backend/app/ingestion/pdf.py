"""PDF parser using PyMuPDF (fitz).

Preserves page numbers so chunks can be traced back to the source page.
"""

from typing import List

import fitz

from app.ingestion.base import DocumentParser, register_parser
from app.models.workflow import DocumentModel


@register_parser("pdf")
class PDFParser(DocumentParser):
    def parse(self, content: bytes, filename: str, document_id: str) -> DocumentModel:
        from app.core.exceptions import MalformedDocumentError

        try:
            doc = fitz.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise MalformedDocumentError(
                f"Could not open PDF '{filename}'."
            ) from exc

        pages: List[str] = []
        sections: List[dict] = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
                sections.append(
                    {
                        "page_number": page_num + 1,
                        "text": text,
                        "section": None,
                    }
                )
        doc.close()

        source_text = "\n\n".join(pages)
        self._empty_check(source_text, filename)

        return DocumentModel(
            document_id=document_id,
            filename=filename,
            file_type="pdf",
            source_text=source_text,
            metadata={"page_count": len(pages)},
            sections=sections,
        )
