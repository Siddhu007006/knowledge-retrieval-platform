"""Persistent document and chunk metadata store.

Uses JSON files on disk so the system survives process restart without
requiring an external database.  The store is a simple key-value mapping
from ``document_id`` to metadata, plus a separate index of chunk_id →
document_id for fast lookups.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

from app.core.config import settings
from app.models.workflow import ChunkModel, DocumentModel


class DocumentStore:
    """File-backed store for document and chunk metadata."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self._dir = base_dir or settings.storage_path()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._docs_file = self._dir / "documents.json"
        self._chunks_file = self._dir / "chunks.json"
        self._docs: Dict[str, dict] = {}
        self._chunks: Dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        self._docs = self._read_json(self._docs_file, "documents store")
        self._chunks = self._read_json(self._chunks_file, "chunks store")

    def _read_json(self, path: Path, label: str) -> dict:
        """Read a JSON file, returning {} on missing, empty, or corrupt files."""
        if not path.exists():
            return {}
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            logger.warning("%s file is empty, resetting to {}: %s", label, path)
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error(
                "Corrupt %s file (JSONDecodeError: %s), resetting to {}: %s",
                label, exc, path,
            )
            # Back up the corrupt file so data is not silently lost
            backup = path.with_suffix(".json.bak")
            path.rename(backup)
            logger.warning("Corrupt file backed up to %s", backup)
            return {}

    def _save(self) -> None:
        """Write JSON atomically using a temp file to avoid partial writes."""
        self._atomic_write(self._docs_file, self._docs)
        self._atomic_write(self._chunks_file, self._chunks)

    def _atomic_write(self, target: Path, data: dict) -> None:
        payload = json.dumps(data, ensure_ascii=False)
        tmp_fd, tmp_path_str = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        tmp_path = Path(tmp_path_str)
        try:
            with open(tmp_fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            tmp_path.replace(target)  # atomic on same filesystem
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    def save_document(
        self,
        document: DocumentModel,
        chunks: List[ChunkModel],
        size_bytes: int,
        created_at: str,
    ) -> None:
        self._docs[document.document_id] = {
            "document_id": document.document_id,
            "filename": document.filename,
            "file_type": document.file_type,
            "chunk_count": len(chunks),
            "size_bytes": size_bytes,
            "created_at": created_at,
        }
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk.model_dump()
        self._save()

    def get_document(self, document_id: str) -> Optional[dict]:
        return self._docs.get(document_id)

    def list_documents(self) -> List[dict]:
        return sorted(
            self._docs.values(), key=lambda d: d.get("created_at", "")
        )

    def get_chunk(self, chunk_id: str) -> Optional[ChunkModel]:
        data = self._chunks.get(chunk_id)
        return ChunkModel(**data) if data else None

    def get_chunks_for_document(self, document_id: str) -> List[ChunkModel]:
        return [
            ChunkModel(**c)
            for c in self._chunks.values()
            if c.get("document_id") == document_id
        ]

    def get_all_chunks(self) -> List[ChunkModel]:
        return [ChunkModel(**c) for c in self._chunks.values()]

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[ChunkModel]:
        return [
            ChunkModel(**self._chunks[cid])
            for cid in chunk_ids
            if cid in self._chunks
        ]

    def delete_document(self, document_id: str) -> bool:
        if document_id not in self._docs:
            return False
        del self._docs[document_id]
        to_remove = [
            cid
            for cid, chunk in self._chunks.items()
            if chunk.get("document_id") == document_id
        ]
        for cid in to_remove:
            del self._chunks[cid]
        self._save()
        return True

    def document_exists(self, document_id: str) -> bool:
        return document_id in self._docs

    def count(self) -> int:
        return len(self._docs)
