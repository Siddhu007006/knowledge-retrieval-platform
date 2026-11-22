
import pytest

from app.retrieval import rate_limiter
from app.retrieval.rate_limiter import (
    EmbeddingRateLimiter,
    EmbeddingQuotaBudgetExceeded,
)


def test_token_estimation(tmp_path):
    limiter = EmbeddingRateLimiter(
        rpm_limit=80,
        tpm_limit=24000,
        rpd_limit=800,
        state_path=tmp_path / "usage.json",
        chars_per_token=4,
    )

    assert limiter.estimate_tokens("12345678") == 2
    assert limiter.estimate_tokens("") == 1


def test_reservations_are_persisted(tmp_path):
    state = tmp_path / "usage.json"

    first = EmbeddingRateLimiter(
        rpm_limit=80,
        tpm_limit=24000,
        rpd_limit=3,
        state_path=state,
    )

    first.wait_for_capacity(2, 100)

    second = EmbeddingRateLimiter(
        rpm_limit=80,
        tpm_limit=24000,
        rpd_limit=3,
        state_path=state,
    )

    with pytest.raises(EmbeddingQuotaBudgetExceeded):
        second.wait_for_capacity(2, 100)


def test_rpm_waits_for_previous_reservations(
    tmp_path,
    monkeypatch,
):
    clock = {"now": 1000.0}

    def fake_time():
        return clock["now"]

    def fake_sleep(seconds):
        clock["now"] += seconds

    monkeypatch.setattr(
        rate_limiter.time,
        "time",
        fake_time,
    )
    monkeypatch.setattr(
        rate_limiter.time,
        "sleep",
        fake_sleep,
    )

    limiter = EmbeddingRateLimiter(
        rpm_limit=2,
        tpm_limit=100,
        rpd_limit=100,
        state_path=tmp_path / "usage.json",
    )

    limiter.wait_for_capacity(2, 20)
    first_request_time = clock["now"]

    limiter.wait_for_capacity(1, 10)

    assert clock["now"] - first_request_time >= 60


def test_tpm_budget_rejects_oversized_batch(tmp_path):
    limiter = EmbeddingRateLimiter(
        rpm_limit=80,
        tpm_limit=100,
        rpd_limit=800,
        state_path=tmp_path / "usage.json",
    )

    with pytest.raises(EmbeddingQuotaBudgetExceeded):
        limiter.wait_for_capacity(1, 101)
