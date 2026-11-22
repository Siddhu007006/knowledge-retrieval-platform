"""Parser interface and shared document model for ingestion."""

from abc import ABC, abstractmethod
from typing import List

from app.models.workflow import DocumentModel


class DocumentParser(ABC):
    """Format-specific parser contract.

    Every parser receives the raw file bytes and filename and returns a
    ``DocumentModel`` with normalized text and metadata.  Metadata such as
    page numbers, headings, or section names is preserved in ``sections``
    so the chunker can produce semantically coherent chunks.
    """

    @abstractmethod
    def parse(self, content: bytes, filename: str, document_id: str) -> DocumentModel:
        ...

    @staticmethod
    def _empty_check(text: str, filename: str) -> None:
        from app.core.exceptions import EmptyFileError

        if not text or not text.strip():
            raise EmptyFileError(f"The file '{filename}' contains no extractable text.")


PARSER_REGISTRY: dict[str, type[DocumentParser]] = {}


def register_parser(file_type: str) -> type:
    """Decorator that registers a parser class for a file extension."""

    def _wrap(cls: type[DocumentParser]) -> type[DocumentParser]:
        PARSER_REGISTRY[file_type.lower()] = cls
        return cls

    return _wrap


def get_parser(file_type: str) -> DocumentParser:
    from app.core.exceptions import UnsupportedFileTypeError

    cls = PARSER_REGISTRY.get(file_type.lower())
    if cls is None:
        raise UnsupportedFileTypeError(
            f"No parser is registered for file type '{file_type}'."
        )
    return cls()


def supported_types() -> List[str]:
    return sorted(PARSER_REGISTRY.keys())
