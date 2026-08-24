"""Process-local operational readiness for the primary LLM provider.

The provider is mandatory at startup, but successful construction proves only
that configuration exists.  This tracker records the outcome of real ``chat``
calls so ``GET /health`` can distinguish configured capability from recent
operational readiness without issuing a billable synthetic probe.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import Lock

from everos.component.utils.datetime import get_utc_now


@dataclass(frozen=True, slots=True)
class LLMReadiness:
    """Immutable snapshot of outcomes observed at the LLM client boundary."""

    healthy: bool | None
    last_success_at: datetime | None
    last_failure_at: datetime | None
    consecutive_failures: int
    last_error_type: str | None


_lock = Lock()
_state = LLMReadiness(
    healthy=None,
    last_success_at=None,
    last_failure_at=None,
    consecutive_failures=0,
    last_error_type=None,
)


def get_llm_readiness() -> LLMReadiness:
    """Return a consistent snapshot without performing an upstream probe."""
    with _lock:
        return _state


def record_llm_success() -> None:
    """Mark a real provider call successful and clear its failure streak."""
    global _state
    with _lock:
        _state = LLMReadiness(
            healthy=True,
            last_success_at=get_utc_now(),
            last_failure_at=_state.last_failure_at,
            consecutive_failures=0,
            last_error_type=None,
        )


def record_llm_failure(error: Exception) -> None:
    """Record a failed real provider call without retaining its message."""
    global _state
    with _lock:
        _state = LLMReadiness(
            healthy=False,
            last_success_at=_state.last_success_at,
            last_failure_at=get_utc_now(),
            consecutive_failures=_state.consecutive_failures + 1,
            # Exception class names are bounded operational metadata.  Messages
            # can contain provider responses or request details and are never
            # retained in the health state.
            last_error_type=type(error).__name__,
        )


def _reset_for_tests() -> None:
    """Reset process state; test-only because production state is monotonic."""
    global _state
    with _lock:
        _state = LLMReadiness(
            healthy=None,
            last_success_at=None,
            last_failure_at=None,
            consecutive_failures=0,
            last_error_type=None,
        )
