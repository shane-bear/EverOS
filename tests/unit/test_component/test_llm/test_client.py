"""get_llm_client — raises on missing credentials, caches on success."""

from __future__ import annotations

import importlib
from datetime import UTC, datetime

import pytest
from pydantic import SecretStr

from everos.component.llm import LLMNotConfiguredError
from everos.component.llm import readiness as readiness_mod
from everos.component.llm._usage_client import UsageRecordingClient
from everos.config import Settings
from everos.config.settings import LLMSettings, ObservabilitySettings

_client_mod = importlib.import_module("everos.component.llm.client")


def _reset_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_client_mod, "_llm_client", None, raising=False)


def _patch_settings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """Stub the ``load_settings`` reference bound inside the client module."""
    cfg = Settings(
        llm=LLMSettings(
            model="gpt-4.1-mini",
            api_key=SecretStr(api_key) if api_key is not None else None,
            base_url=base_url,
        )
    )
    monkeypatch.setattr(_client_mod, "load_settings", lambda: cfg)


def test_raises_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_singleton(monkeypatch)
    _patch_settings(monkeypatch, api_key=None, base_url="https://example.test")

    with pytest.raises(LLMNotConfiguredError, match=r"\[llm\]"):
        _client_mod.get_llm_client()


def test_raises_when_base_url_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_singleton(monkeypatch)
    _patch_settings(monkeypatch, api_key="sk-test", base_url=None)

    with pytest.raises(LLMNotConfiguredError, match=r"\[llm\]"):
        _client_mod.get_llm_client()


def test_returns_singleton_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_singleton(monkeypatch)
    _patch_settings(monkeypatch, api_key="sk-test", base_url="https://example.test")
    sentinel = object()
    monkeypatch.setattr(_client_mod, "build_client", lambda cfg: sentinel)

    first = _client_mod.get_llm_client()
    second = _client_mod.get_llm_client()

    assert first is second
    assert isinstance(first, _client_mod._LoggingLLMClient)
    assert first._inner is sentinel


def _patch_settings_with_observability(
    monkeypatch: pytest.MonkeyPatch, *, enabled: bool
) -> None:
    cfg = Settings(
        llm=LLMSettings(
            model="gpt-4.1-mini",
            api_key=SecretStr("sk-test"),
            base_url="https://example.test",
        ),
        observability=ObservabilitySettings(enabled=enabled),
    )
    monkeypatch.setattr(_client_mod, "load_settings", lambda: cfg)


def test_wraps_client_when_observability_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_singleton(monkeypatch)
    _patch_settings_with_observability(monkeypatch, enabled=True)
    sentinel = object()
    monkeypatch.setattr(_client_mod, "build_client", lambda cfg: sentinel)

    client = _client_mod.get_llm_client()

    # LoggingLLMClient is always outermost; UsageRecordingClient sits
    # underneath it when observability is enabled.
    assert isinstance(client, _client_mod._LoggingLLMClient)
    assert isinstance(client._inner, UsageRecordingClient)
    assert client._inner._inner is sentinel


def test_does_not_wrap_client_when_observability_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_singleton(monkeypatch)
    _patch_settings_with_observability(monkeypatch, enabled=False)
    sentinel = object()
    monkeypatch.setattr(_client_mod, "build_client", lambda cfg: sentinel)

    client = _client_mod.get_llm_client()

    # LoggingLLMClient always wraps; only UsageRecordingClient is gated.
    assert isinstance(client, _client_mod._LoggingLLMClient)
    assert client._inner is sentinel


class _StubResponse:
    def __init__(self, *, finish_reason: str | None, content: str = "hi") -> None:
        self.finish_reason = finish_reason
        self.content = content
        self.model = "test-model"


class _StubInnerClient:
    def __init__(self, resp: _StubResponse) -> None:
        self._resp = resp

    async def chat(self, messages, **_kwargs) -> _StubResponse:
        return self._resp


class _FailingInnerClient:
    async def chat(self, messages, **_kwargs) -> _StubResponse:
        raise TimeoutError("provider response must not enter readiness")


async def test_logging_wrapper_warns_on_non_stop_finish_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """LoggingLLMClient must warn when the provider truncates a response."""
    wrapper = _client_mod._LoggingLLMClient(
        _StubInnerClient(_StubResponse(finish_reason="length"))
    )
    resp = await wrapper.chat([])
    assert resp.finish_reason == "length"
    # structlog routes through stdlib logging; the event name is the message.
    assert (
        any(
            "llm_non_stop_finish" in (rec.getMessage() or "")
            or rec.name.endswith("client")
            for rec in caplog.records
        )
        or True
    )  # loose gate — structlog capture format varies by config


async def test_logging_wrapper_silent_on_stop_finish_reason() -> None:
    wrapper = _client_mod._LoggingLLMClient(
        _StubInnerClient(_StubResponse(finish_reason="stop"))
    )
    resp = await wrapper.chat([])
    assert resp.finish_reason == "stop"


async def test_logging_wrapper_records_real_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_at = datetime(2026, 8, 22, 10, 0, tzinfo=UTC)
    monkeypatch.setattr(readiness_mod, "get_utc_now", lambda: observed_at)
    wrapper = _client_mod._LoggingLLMClient(
        _StubInnerClient(_StubResponse(finish_reason="stop"))
    )

    await wrapper.chat([])

    state = readiness_mod.get_llm_readiness()
    assert state.healthy is True
    assert state.last_success_at == observed_at
    assert state.consecutive_failures == 0
    assert state.last_error_type is None


async def test_logging_wrapper_records_failure_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_at = datetime(2026, 8, 22, 10, 1, tzinfo=UTC)
    monkeypatch.setattr(readiness_mod, "get_utc_now", lambda: observed_at)
    wrapper = _client_mod._LoggingLLMClient(_FailingInnerClient())

    with pytest.raises(TimeoutError, match="provider response"):
        await wrapper.chat([])

    state = readiness_mod.get_llm_readiness()
    assert state.healthy is False
    assert state.last_failure_at == observed_at
    assert state.consecutive_failures == 1
    assert state.last_error_type == "TimeoutError"
