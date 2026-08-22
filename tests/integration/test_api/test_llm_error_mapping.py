"""Real memory routes map exhausted EverAlgo LLM failures to safe HTTP 503."""

from __future__ import annotations

import importlib

import pytest
from everalgo.llm import LLMError
from httpx import ASGITransport, AsyncClient

from everos.entrypoints.api.app import create_app

_memorize_routes = importlib.import_module("everos.entrypoints.api.routes.memorize")


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        pytest.param(
            "/api/v2/memory/add",
            {
                "session_id": "session-1",
                "messages": [
                    {
                        "sender_id": "user-1",
                        "role": "user",
                        "timestamp": 1_700_000_000_000,
                        "content": "hello",
                    }
                ],
            },
            id="add",
        ),
        pytest.param(
            "/api/v2/memory/flush",
            {"session_id": "session-1"},
            id="flush",
        ),
    ],
)
async def test_memory_route_returns_dedicated_retryable_error(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    payload: dict,
) -> None:
    async def _fail(*args, **kwargs):
        raise LLMError("provider detail must remain server-side")

    monkeypatch.setattr(_memorize_routes, "memorize", _fail)
    app = create_app()
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(path, json=payload)

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "UPSTREAM_LLM_UNAVAILABLE"
    assert body["error"]["message"] == (
        "Upstream LLM request failed; retry may succeed."
    )
    assert "provider detail" not in str(body)
