"""Parser tests for the four required ingestion formats."""
from io import BytesIO

import pytest
from docx import Document
import fitz

from app.core.exceptions import EmptyFileError, MalformedDocumentError
from app.ingestion.csv import CSVParser
from app.ingestion.docx import DOCXParser
from app.ingestion.pdf import PDFParser
from app.ingestion.txt import TXTParser


class TestTXTParser:
    def test_parse_basic_and_paragraph_boundaries(self):
        doc = TXTParser().parse(b"First paragraph.\n\nSecond paragraph.", "test.txt", "doc-1")
        assert doc.file_type == "txt"
        assert len(doc.sections) == 2

    def test_empty_file_rejected(self):
        with pytest.raises(EmptyFileError):
            TXTParser().parse(b"   ", "empty.txt", "doc-1")


class TestCSVParser:
    def test_headered_csv_maps_cells_to_column_names(self):
        doc = CSVParser().parse(b"name,department\nAsha,Engineering\nRavi,Finance", "staff.csv", "doc-2")
        assert doc.file_type == "csv"
        assert doc.metadata["row_count"] == 2
        assert "department: Engineering" in doc.sections[0]["text"]

    def test_utf8_bom_is_supported(self):
        doc = CSVParser().parse("name,role\nAsha,Analyst".encode("utf-8-sig"), "staff.csv", "doc-3")
        assert "Asha" in doc.source_text


class TestDOCXParser:
    def test_paragraph_and_table_extraction(self):
        source = Document()
        source.add_heading("IT policy", level=1)
        source.add_paragraph("Use the approved VPN client.")
        table = source.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "System"
        table.cell(0, 1).text = "Owner"
        table.cell(1, 0).text = "VPN"
        table.cell(1, 1).text = "IT"
        buffer = BytesIO()
        source.save(buffer)

        doc = DOCXParser().parse(buffer.getvalue(), "policy.docx", "doc-4")
        assert doc.file_type == "docx"
        assert "approved VPN" in doc.source_text
        assert "Owner: IT" in doc.source_text
        assert doc.metadata["table_count"] == 1

    def test_invalid_docx_rejected(self):
        with pytest.raises(MalformedDocumentError):
            DOCXParser().parse(b"not docx", "bad.docx", "doc-5")


class TestPDFParser:
    def test_extracts_text_and_page_number(self):
        source = fitz.open()
        page = source.new_page()
        page.insert_text((72, 72), "PDF ingestion validation content")
        content = source.tobytes()
        source.close()
        doc = PDFParser().parse(content, "sample.pdf", "doc-pdf")
        assert doc.file_type == "pdf"
        assert "PDF ingestion validation" in doc.source_text
        assert doc.sections[0]["page_number"] == 1

    def test_parse_corrupt_pdf(self):
        with pytest.raises(MalformedDocumentError):
            PDFParser().parse(b"not a real pdf", "bad.pdf", "doc-6")

    def test_empty_pdf_rejected(self):
        with pytest.raises((EmptyFileError, MalformedDocumentError)):
            PDFParser().parse(b"", "empty.pdf", "doc-7")
