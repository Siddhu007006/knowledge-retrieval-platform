"""Structured JSON logging configured once at import time."""

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from app.core.config import settings

_LOG_FIELDS = (
    "timestamp",
    "level",
    "logger",
    "message",
    "request_id",
    "document_id",
    "query_id",
    "stage",
    "latency_ms",
    "candidate_count",
    "reranked_count",
    "selected_count",
    "grounding_status",
    "retry_count",
    "error",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in _LOG_FIELDS:
            if key in payload:
                continue
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _configure() -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    return root


logger = _configure()


def new_request_id() -> str:
    return str(uuid.uuid4())


_RESERVED = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
})


def log_event(
    message: str,
    *,
    level: str = "INFO",
    request_id: Optional[str] = None,
    document_id: Optional[str] = None,
    query_id: Optional[str] = None,
    stage: Optional[str] = None,
    **extra: Any,
) -> None:
    """Emit one structured log record with arbitrary extra fields.

    Fields that collide with reserved ``LogRecord`` attributes are prefixed
    with ``ctx_`` to avoid ``KeyError`` from the logging framework.
    """
    kwargs: dict[str, Any] = {}
    if request_id:
        kwargs["request_id"] = request_id
    if document_id:
        kwargs["document_id"] = document_id
    if query_id:
        kwargs["query_id"] = query_id
    if stage:
        kwargs["stage"] = stage
    for key, val in extra.items():
        safe_key = f"ctx_{key}" if key in _RESERVED else key
        kwargs[safe_key] = val
    getattr(logger, level.lower(), logger.info)(message, extra=kwargs)
