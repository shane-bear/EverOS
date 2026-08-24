"""Operational LLM readiness state transitions and safe retained metadata."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from everos.component.llm import readiness


def test_initial_state_is_unknown_not_a_claim_of_provider_health() -> None:
    state = readiness.get_llm_readiness()
    assert state.healthy is None
    assert state.last_success_at is None
    assert state.last_failure_at is None
    assert state.consecutive_failures == 0
    assert state.last_error_type is None


def test_failures_accumulate_without_retaining_sensitive_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = datetime(2026, 8, 22, 10, 0, tzinfo=UTC)
    second = datetime(2026, 8, 22, 10, 1, tzinfo=UTC)
    times = iter([first, second])
    monkeypatch.setattr(readiness, "get_utc_now", lambda: next(times))

    readiness.record_llm_failure(TimeoutError("secret provider response"))
    readiness.record_llm_failure(ConnectionError("https://key@example.test"))

    state = readiness.get_llm_readiness()
    assert state.healthy is False
    assert state.last_failure_at == second
    assert state.consecutive_failures == 2
    assert state.last_error_type == "ConnectionError"
    assert "secret" not in repr(state)
    assert "example.test" not in repr(state)


def test_success_clears_streak_but_preserves_last_failure_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_at = datetime(2026, 8, 22, 10, 0, tzinfo=UTC)
    succeeded_at = datetime(2026, 8, 22, 10, 2, tzinfo=UTC)
    times = iter([failed_at, succeeded_at])
    monkeypatch.setattr(readiness, "get_utc_now", lambda: next(times))

    readiness.record_llm_failure(TimeoutError())
    readiness.record_llm_success()

    state = readiness.get_llm_readiness()
    assert state.healthy is True
    assert state.last_success_at == succeeded_at
    assert state.last_failure_at == failed_at
    assert state.consecutive_failures == 0
    assert state.last_error_type is None
