"""Health check route."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel

from everos import __version__
from everos.component.capabilities import compute_disabled_features
from everos.component.embedding import get_embedding_capability
from everos.component.llm import get_llm_readiness
from everos.component.multimodal import get_multimodal_llm_capability
from everos.component.parser import parser_available
from everos.component.rerank import get_rerank_capability
from everos.core.observability.logging import get_logger
from everos.entrypoints.api.utils import cascade_orchestrator

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


class HealthCapabilities(BaseModel):
    """Availability flags for the five capability probes.

    Field order matches the health-endpoint payload contract; clients
    key off these names to decide whether to expose optional features.
    """

    llm: bool
    embed: bool
    rerank: bool
    multimodal_llm: bool
    parser: bool


class CascadeHealthBlock(BaseModel):
    """Readiness of the md → LanceDB projection (cascade) subsystem.

    ``healthy`` reflects **operational** health only — drain loop alive,
    optimize not stuck, version cleanup (prune) not stalled — and is what
    alerting should watch. ``failed_permanent`` (md files awaiting
    ``cascade fix``) is a normal data-quality backlog reported as an
    informational count; it does **not** flip ``healthy``, otherwise the
    signal would sit red forever.
    """

    healthy: bool
    reasons: list[str]
    pending: int
    failed_permanent: int
    failed_retryable: int
    drain_consecutive_failures: int
    unrecoverable_total: int
    optimize_failure_streak: int
    prune_stale_seconds: float


class LLMReadinessBlock(BaseModel):
    """Outcomes observed from real calls to the primary LLM provider.

    ``healthy=None`` means no call has completed since this process started.
    The block never performs a synthetic provider call, so reading health is
    fast and cannot spend tokens.
    """

    healthy: bool | None
    last_success_at: datetime | None
    last_failure_at: datetime | None
    consecutive_failures: int
    last_error_type: str | None


class HealthResponse(BaseModel):
    """Response schema for ``GET /health``.

    Declared as a Pydantic model (not ``dict``) so the generated
    OpenAPI schema carries the full field shape — ``capabilities``,
    ``disabled_features`` and ``cascade`` are typed. A bare ``-> dict``
    return type degrades the OpenAPI response to
    ``additionalProperties: true``, which robs clients (and codegen) of
    any structure to lean on.
    """

    status: str
    version: str
    capabilities: HealthCapabilities
    disabled_features: list[str]
    llm_readiness: LLMReadinessBlock
    cascade: CascadeHealthBlock | None = None
    """Present when the cascade lifespan is running; ``None`` for a
    minimal app built without it."""


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness + capabilities + cascade readiness probe.

    ``status`` stays ``"ok"`` whenever the process is up — the HTTP code
    is a *liveness* signal and a degraded cascade must not trigger a
    restart (a crash-loop fixes neither a bad md file nor disk bloat).
    The ``llm_readiness`` block reports outcomes from real provider calls;
    ``healthy=None`` means no call has completed in this process.  It performs
    no synthetic probe.  The ``cascade`` block is its subsystem readiness
    signal: ``healthy=false``
    with human-readable ``reasons`` **only** when the projection pipeline
    itself is stuck (drain failing, optimize stuck, version cleanup
    stalled). ``failed_permanent`` — files awaiting ``cascade fix`` — is
    a data-quality backlog reported as an informational count that does
    not flip ``healthy``. Alert on ``cascade.healthy``.
    """
    # ``capabilities.llm`` means configuration passed the mandatory startup
    # gate; it is not a reachability claim.  Real call outcomes live in the
    # separate ``llm_readiness`` block so liveness stays independent from an
    # external provider.
    caps = HealthCapabilities(
        llm=True,
        embed=get_embedding_capability().available,
        rerank=get_rerank_capability().available,
        multimodal_llm=get_multimodal_llm_capability().available,
        parser=parser_available(),
    )
    llm = get_llm_readiness()
    cascade: CascadeHealthBlock | None = None
    orch = cascade_orchestrator(request)
    if orch is not None:
        try:
            ch = await orch.health()
            cascade = CascadeHealthBlock(
                healthy=ch.healthy,
                reasons=ch.reasons,
                pending=ch.pending,
                failed_permanent=ch.failed_permanent,
                failed_retryable=ch.failed_retryable,
                drain_consecutive_failures=ch.drain_consecutive_failures,
                unrecoverable_total=ch.unrecoverable_total,
                optimize_failure_streak=ch.optimize_failure_streak,
                prune_stale_seconds=round(ch.prune_stale_seconds, 1),
            )
        except Exception as exc:
            # The probe reads SQLite (queue_summary runs aggregate counts).
            # A locked / full / mid-migration DB must NOT turn /health into a
            # 500 — that flips the liveness signal and makes k8s restart the
            # container, which fixes neither a stuck DB nor disk bloat (see
            # the handler docstring). Surface it as unhealthy *readiness*
            # with a reason and keep HTTP 200.
            logger.warning("cascade_health_probe_failed", error=repr(exc))
            cascade = CascadeHealthBlock(
                healthy=False,
                reasons=[f"cascade health probe failed: {exc!r}"],
                pending=0,
                failed_permanent=0,
                failed_retryable=0,
                drain_consecutive_failures=0,
                unrecoverable_total=0,
                optimize_failure_streak=0,
                prune_stale_seconds=0.0,
            )
    return HealthResponse(
        status="ok",
        version=__version__,
        capabilities=caps,
        disabled_features=compute_disabled_features(caps.model_dump()),
        llm_readiness=LLMReadinessBlock(
            healthy=llm.healthy,
            last_success_at=llm.last_success_at,
            last_failure_at=llm.last_failure_at,
            consecutive_failures=llm.consecutive_failures,
            last_error_type=llm.last_error_type,
        ),
        cascade=cascade,
    )
