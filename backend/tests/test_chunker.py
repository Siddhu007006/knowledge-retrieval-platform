"""Unit tests for the chunker."""

from app.ingestion.chunker import chunk_document
from app.models.workflow import DocumentModel


def _make_doc(text, sections=None, file_type="txt"):
    return DocumentModel(
        document_id="doc-1",
        filename="test.txt",
        file_type=file_type,
        source_text=text,
        sections=sections or [{"page_number": None, "text": text, "section": None}],
    )


class TestChunker:
    def test_short_text_one_chunk(self):
        doc = _make_doc("Hello world.")
        chunks = chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."
        assert chunks[0].document_id == "doc-1"
        assert chunks[0].chunk_id  # UUID generated

    def test_long_text_multiple_chunks(self, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.chunk_size", 50)
        monkeypatch.setattr("app.core.config.settings.chunk_overlap", 10)
        long_text = "\n\n".join([f"Paragraph {i}. " * 20 for i in range(10)])
        doc = _make_doc(long_text)
        chunks = chunk_document(doc)
        assert len(chunks) > 1

    def test_preserves_section_metadata(self):
        sections = [
            {"page_number": 5, "text": "Page five content.", "section": "Intro"},
            {"page_number": 6, "text": "Page six content.", "section": "Body"},
        ]
        doc = _make_doc("combined", sections=sections, file_type="pdf")
        chunks = chunk_document(doc)
        assert any(c.page_number == 5 for c in chunks)
        assert any(c.section == "Intro" for c in chunks)
        assert any(c.section == "Body" for c in chunks)

    def test_chunk_has_metadata(self):
        doc = _make_doc("Some text here.")
        chunks = chunk_document(doc)
        assert "estimated_tokens" in chunks[0].metadata
        assert "source_text_length" in chunks[0].metadata

    def test_empty_sections_skipped(self):
        sections = [
            {"page_number": None, "text": "", "section": None},
            {"page_number": None, "text": "Real content.", "section": None},
        ]
        doc = _make_doc("combined", sections=sections)
        chunks = chunk_document(doc)
        assert len(chunks) == 1
        assert "Real content" in chunks[0].text

    def test_csv_rows_remain_independent_chunks(self):
        sections = [
            {
                "page_number": None,
                "text": "employee_id: E001 | name: Asha | department: Engineering",
                "section": "Row 2",
            },
            {
                "page_number": None,
                "text": "employee_id: E002 | name: Rahul | department: HR",
                "section": "Row 3",
            },
            {
                "page_number": None,
                "text": "employee_id: E003 | name: Meera | department: Finance",
                "section": "Row 4",
            },
        ]

        doc = _make_doc(
            "CSV test data",
            sections=sections,
            file_type="csv",
        )

        chunks = chunk_document(doc)

        assert len(chunks) == 3

        assert "name: Asha" in chunks[0].text
        assert "name: Rahul" not in chunks[0].text

        assert "name: Rahul" in chunks[1].text
        assert "name: Asha" not in chunks[1].text
        assert "name: Meera" not in chunks[1].text

        assert "name: Meera" in chunks[2].text
        assert "name: Rahul" not in chunks[2].text

        assert [chunk.section for chunk in chunks] == [
            "Row 2", "Row 3", "Row 4"
        ]
