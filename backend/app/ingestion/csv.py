"""CSV text extraction; each data row is rendered with its column labels."""
import csv
from io import StringIO

from app.ingestion.base import DocumentParser, register_parser
from app.models.workflow import DocumentModel


@register_parser("csv")
class CSVParser(DocumentParser):
    def parse(self, content: bytes, filename: str, document_id: str) -> DocumentModel:
        from app.core.exceptions import MalformedDocumentError

        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = content.decode("latin-1", errors="replace")

        try:
            rows = list(csv.reader(StringIO(text)))
        except csv.Error as exc:
            raise MalformedDocumentError(f"Could not parse CSV '{filename}'.") from exc

        rows = [row for row in rows if any(cell.strip() for cell in row)]
        self._empty_check(text, filename)
        if not rows:
            self._empty_check("", filename)

        headers = [cell.strip() or f"Column {i + 1}" for i, cell in enumerate(rows[0])]
        sections: list[dict] = []
        rendered: list[str] = []
        for row_number, row in enumerate(rows[1:], start=2):
            values = [cell.strip() for cell in row]
            fields = []
            for index, header in enumerate(headers):
                value = values[index] if index < len(values) else ""
                if value:
                    fields.append(f"{header}: {value}")
            if fields:
                row_text = " | ".join(fields)
                rendered.append(row_text)
                sections.append({"page_number": None, "text": row_text, "section": f"Row {row_number}"})

        # Preserve header-only CSVs so ingestion can be validated, while avoiding empty chunks.
        if not rendered:
            header_text = "CSV columns: " + ", ".join(headers)
            rendered = [header_text]
            sections = [{"page_number": None, "text": header_text, "section": "Header"}]

        return DocumentModel(
            document_id=document_id,
            filename=filename,
            file_type="csv",
            source_text="\n\n".join(rendered),
            metadata={"row_count": max(0, len(rows) - 1), "column_count": len(headers), "columns": headers},
            sections=sections,
        )
