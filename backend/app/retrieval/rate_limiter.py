
"""Conservative, persistent quota management for Gemini embeddings.

Counts each text input as one RPM/RPD unit. Uses an estimated token
count for TPM and persists reservations across application restarts.

Designed for a single local backend process. Use a distributed limiter
such as Redis if the application runs multiple workers or instances.
"""

import json
import logging
import math
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Tuple

logger = logging.getLogger("app.embedding_rate_limiter")

UsageEntry = Tuple[float, int, int]


class EmbeddingQuotaBudgetExceeded(RuntimeError):
    """Raised when the local rolling 24-hour budget is exhausted."""


class EmbeddingRateLimiter:
    MINUTE_WINDOW_SECONDS = 60.0
    DAILY_WINDOW_SECONDS = 86400.0

    def __init__(
        self,
        rpm_limit: int,
        tpm_limit: int,
        rpd_limit: int,
        state_path: Path,
        chars_per_token: float = 3.5,
    ) -> None:
        if min(rpm_limit, tpm_limit, rpd_limit) <= 0:
            raise ValueError("Embedding quota limits must be positive.")

        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive.")

        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.rpd_limit = rpd_limit
        self.chars_per_token = chars_per_token
        self.state_path = Path(state_path)

        self._entries: Deque[UsageEntry] = deque()
        self._lock = threading.Lock()

        self._load_state()

    def estimate_tokens(self, text: str) -> int:
        """Conservative local estimate; not an exact model tokenizer."""
        return max(1, math.ceil(len(text) / self.chars_per_token))

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return

        try:
            payload = json.loads(
                self.state_path.read_text(encoding="utf-8")
            )

            for item in payload.get("entries", []):
                timestamp = float(item["timestamp"])
                input_count = int(item["input_count"])
                token_count = int(item["token_count"])

                if input_count > 0 and token_count > 0:
                    self._entries.append(
                        (timestamp, input_count, token_count)
                    )

            self._prune(time.time())

        except (OSError, ValueError, TypeError, KeyError) as exc:
            # Fail closed: do not silently forget quota usage.
            raise RuntimeError(
                f"Cannot safely load embedding usage state: "
                f"{self.state_path}"
            ) from exc

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "entries": [
                {
                    "timestamp": timestamp,
                    "input_count": input_count,
                    "token_count": token_count,
                }
                for timestamp, input_count, token_count in self._entries
            ]
        }

        temporary_path = self.state_path.with_name(
            self.state_path.name + ".tmp"
        )

        temporary_path.write_text(
            json.dumps(payload, separators=(",", ":")),
            encoding="utf-8",
        )

        os.replace(temporary_path, self.state_path)

    def _prune(self, now: float) -> None:
        while (
            self._entries
            and now - self._entries[0][0] >= self.DAILY_WINDOW_SECONDS
        ):
            self._entries.popleft()

    def wait_for_capacity(
        self,
        input_count: int,
        estimated_tokens: int,
    ) -> None:
        """Reserve quota before sending an embedding request.

        Failed attempts remain reserved conservatively, because a provider
        may count requests even when it returns a rate-limit response.
        """

        if input_count <= 0 or estimated_tokens <= 0:
            raise ValueError("Input count and tokens must be positive.")

        if input_count > self.rpm_limit:
            raise EmbeddingQuotaBudgetExceeded(
                f"Batch has {input_count} inputs; local RPM target is "
                f"{self.rpm_limit}. Split the batch."
            )

        if estimated_tokens > self.tpm_limit:
            raise EmbeddingQuotaBudgetExceeded(
                f"Batch has an estimated {estimated_tokens} tokens; "
                f"local TPM target is {self.tpm_limit}. Split the batch."
            )

        while True:
            with self._lock:
                now = time.time()
                self._prune(now)

                used_daily_inputs = sum(
                    count for _, count, _ in self._entries
                )

                if used_daily_inputs + input_count > self.rpd_limit:
                    raise EmbeddingQuotaBudgetExceeded(
                        "Local rolling 24-hour embedding budget exhausted: "
                        f"{used_daily_inputs}/{self.rpd_limit} inputs used. "
                        "Wait for previous reservations to expire or "
                        "review the project quota."
                    )

                minute_entries = [
                    entry
                    for entry in self._entries
                    if now - entry[0] < self.MINUTE_WINDOW_SECONDS
                ]

                used_rpm = sum(
                    count for _, count, _ in minute_entries
                )
                used_tpm = sum(
                    tokens for _, _, tokens in minute_entries
                )

                rpm_available = (
                    used_rpm + input_count <= self.rpm_limit
                )
                tpm_available = (
                    used_tpm + estimated_tokens <= self.tpm_limit
                )

                if rpm_available and tpm_available:
                    self._entries.append(
                        (now, input_count, estimated_tokens)
                    )

                    # Persist before making the external API request.
                    self._save_state()

                    logger.info(
                        "Embedding quota reserved: inputs=%d, "
                        "estimated_tokens=%d, rolling_24h=%d/%d",
                        input_count,
                        estimated_tokens,
                        used_daily_inputs + input_count,
                        self.rpd_limit,
                    )
                    return

                remaining_rpm = used_rpm
                remaining_tpm = used_tpm
                wait_seconds = None

                # Find the earliest time enough old reservations expire.
                for timestamp, count, tokens in minute_entries:
                    remaining_rpm -= count
                    remaining_tpm -= tokens

                    if (
                        remaining_rpm + input_count <= self.rpm_limit
                        and remaining_tpm + estimated_tokens
                        <= self.tpm_limit
                    ):
                        wait_seconds = max(
                            0.1,
                            timestamp
                            + self.MINUTE_WINDOW_SECONDS
                            - now
                            + 0.1,
                        )
                        break

                if wait_seconds is None:
                    wait_seconds = self.MINUTE_WINDOW_SECONDS

            logger.warning(
                "Embedding quota pacing: waiting %.1f seconds",
                wait_seconds,
            )
            time.sleep(wait_seconds)
