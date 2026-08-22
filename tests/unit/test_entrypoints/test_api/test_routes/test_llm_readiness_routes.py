"""GET /health exposes real LLM outcomes without changing liveness."""

from __future__ import annotations

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from everos.component.llm import readiness
from everos.entrypoints.api.routes.health import router as health_router


async def _health() -> dict:
    app = FastAPI()
    app.include_router(health_router)
    app.state.lifespan_data = {}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    return response.json()


async def test_health_starts_with_unknown_llm_readiness() -> None:
    body = await _health()
    assert body["status"] == "ok"
    assert body["capabilities"]["llm"] is True
    assert body["llm_readiness"] == {
        "healthy": None,
        "last_success_at": None,
        "last_failure_at": None,
        "consecutive_failures": 0,
        "last_error_type": None,
    }


async def test_provider_failure_degrades_readiness_not_liveness() -> None:
    readiness.record_llm_failure(TimeoutError("must not be returned"))
    readiness.record_llm_failure(ConnectionError("must not be returned"))

    body = await _health()

    assert body["status"] == "ok"
    block = body["llm_readiness"]
    assert block["healthy"] is False
    assert block["consecutive_failures"] == 2
    assert block["last_failure_at"] is not None
    assert block["last_error_type"] == "ConnectionError"
    assert "must not be returned" not in str(block)


async def test_provider_recovery_clears_failure_streak() -> None:
    readiness.record_llm_failure(TimeoutError())
    readiness.record_llm_success()

    body = await _health()

    block = body["llm_readiness"]
    assert block["healthy"] is True
    assert block["last_success_at"] is not None
    assert block["last_failure_at"] is not None
    assert block["consecutive_failures"] == 0
    assert block["last_error_type"] is None
