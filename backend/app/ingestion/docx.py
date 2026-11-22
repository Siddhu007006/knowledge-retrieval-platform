"""DOCX text extraction with paragraph/table structure preservation."""
from io import BytesIO
from docx import Document

from app.ingestion.base import DocumentParser, register_parser
from app.models.workflow import DocumentModel


@register_parser("docx")
class DOCXParser(DocumentParser):
    def parse(self, content: bytes, filename: str, document_id: str) -> DocumentModel:
        from app.core.exceptions import MalformedDocumentError

        try:
            source = Document(BytesIO(content))
        except Exception as exc:
            raise MalformedDocumentError(f"Could not open DOCX '{filename}'.") from exc

        sections: list[dict] = []
        blocks: list[str] = []
        for paragraph in source.paragraphs:
            text = paragraph.text.strip()
            if text:
                blocks.append(text)
                sections.append({"page_number": None, "text": text, "section": paragraph.style.name if paragraph.style else None})

        table_rows = 0
        for table_index, table in enumerate(source.tables, start=1):
            table_data = [
                [cell.text.strip().replace("\n", " ") for cell in row.cells]
                for row in table.rows
            ]
            headers = table_data[0] if table_data else []
            if len(table_data) > 1 and any(headers):
                header_text = "Table columns: " + ", ".join(value or f"Column {index + 1}" for index, value in enumerate(headers))
                blocks.append(header_text)
                sections.append({"page_number": None, "text": header_text, "section": f"Table {table_index}, header"})
            for row_index, cells in enumerate(table_data[1:], start=2):
                fields = []
                for column_index, value in enumerate(cells):
                    if not value:
                        continue
                    label = headers[column_index] if column_index < len(headers) and headers[column_index] else f"Column {column_index + 1}"
                    fields.append(f"{label}: {value}")
                if fields:
                    row_text = " | ".join(fields)
                    blocks.append(row_text)
                    sections.append({"page_number": None, "text": row_text, "section": f"Table {table_index}, row {row_index}"})
                    table_rows += 1
            # Preserve a header-only table as searchable content.
            if len(table_data) == 1 and any(headers):
                row_text = "Table columns: " + ", ".join(headers)
                blocks.append(row_text)
                sections.append({"page_number": None, "text": row_text, "section": f"Table {table_index}, header"})

        source_text = "\n\n".join(blocks)
        self._empty_check(source_text, filename)
        return DocumentModel(
            document_id=document_id,
            filename=filename,
            file_type="docx",
            source_text=source_text,
            metadata={"paragraph_count": len(source.paragraphs), "table_count": len(source.tables), "table_row_count": table_rows},
            sections=sections,
        )
