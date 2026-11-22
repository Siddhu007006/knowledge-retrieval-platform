"""Document ingestion parsers registered by file extension."""
from app.ingestion.base import DocumentParser, get_parser, register_parser, supported_types
from app.ingestion.pdf import PDFParser
from app.ingestion.txt import TXTParser
from app.ingestion.docx import DOCXParser
from app.ingestion.csv import CSVParser

__all__ = ["DocumentParser", "get_parser", "register_parser", "supported_types", "PDFParser", "TXTParser", "DOCXParser", "CSVParser"]
