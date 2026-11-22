
"""Bounded retries for transient Google API errors."""

import email.utils
import logging
import random
import re
import time
from datetime import datetime, timezone
from typing import Callable, TypeVar

T = TypeVar("T")
logger = logging.getLogger("app.retry")

MAX_ATTEMPTS = 4  # One initial attempt plus three retries.
BASE_DELAY = 2.0
MAX_BACKOFF = 30.0
MAX_PROVIDER_WAIT = 300.0


def _extract_retry_seconds(exc: Exception) -> float | None:
    """Read Retry-After header or Google's RetryInfo when available."""

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}

    retry_after = headers.get("Retry-After")

    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            try:
                retry_date = email.utils.parsedate_to_datetime(
                    retry_after
                )
                if retry_date.tzinfo is None:
                    retry_date = retry_date.replace(tzinfo=timezone.utc)

                return max(
                    0.0,
                    (retry_date - datetime.now(timezone.utc)).total_seconds(),
                )
            except (TypeError, ValueError, OverflowError):
                pass

    message = str(exc)

    patterns = (
        r'"retryDelay"\s*:\s*"([\d.]+)s"',
        r"retry(?:ing)?\s+(?:in|after)\s+([\d.]+)\s*s",
    )

    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return float(match.group(1)) + 0.5

    return None


def _is_transient(exc: Exception) -> bool:
    code = getattr(exc, "code", None)

    if code is not None:
        try:
            if int(code) in (408, 429, 500, 502, 503, 504):
                return True
        except (TypeError, ValueError):
            pass

    message = str(exc).lower()

    return any(
        token in message
        for token in (
            "429",
            "500",
            "502",
            "503",
            "504",
            "rate limit",
            "resource exhausted",
            "unavailable",
            "timeout",
            "deadline exceeded",
        )
    )


def with_retry(fn: Callable[[], T]) -> T:
    """Retry transient failures with bounded exponential backoff.

    The wrapped function should perform its own quota reservation
    immediately before every provider request.
    """

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn()

        except Exception as exc:
            if not _is_transient(exc) or attempt == MAX_ATTEMPTS:
                raise

            provider_wait = _extract_retry_seconds(exc)

            if provider_wait is not None:
                # Do not retry early when Google asks for a long wait.
                if provider_wait > MAX_PROVIDER_WAIT:
                    logger.warning(
                        "Provider requested %.1f seconds before retry; "
                        "propagating error instead of blocking the request.",
                        provider_wait,
                    )
                    raise

                delay = provider_wait
            else:
                delay = min(
                    MAX_BACKOFF,
                    BASE_DELAY * (2 ** (attempt - 1)),
                )
                delay *= random.uniform(0.8, 1.2)

            logger.warning(
                "Transient provider error; attempt %d/%d, "
                "retrying in %.1f seconds: %s",
                attempt,
                MAX_ATTEMPTS,
                delay,
                exc,
            )

            time.sleep(delay)

    raise RuntimeError("Unreachable retry state.")
