from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from deepequity.agents import cache
from deepequity.agents.routing import AgentRole, model_for
from deepequity.api.auth import require_auth
from deepequity.core.config import get_settings
from deepequity.core.costs import price_for

router = APIRouter(tags=["stats"])


class CacheStats(BaseModel):
    enabled: bool
    threshold: float
    hits: int
    misses: int
    lookups: int
    hit_rate: float
    tokens_saved: int


class RoleRouting(BaseModel):
    role: str
    model: str
    input_per_million_usd: float
    output_per_million_usd: float
    cacheable: bool


class StatsResponse(BaseModel):
    cache: CacheStats
    routing: list[RoleRouting]


#shows what the cost machinery is actually doing.
#
#the plan asks for a measured cache hit rate and for costs to be exposed rather than
#logged and forgotten, and this is where both live. the routing table is here too because
#the cheapest possible bug in this system is a role quietly pointing at the expensive
#model: nothing breaks, the notes look fine, and the bill is three times what it should
#be. one endpoint that says which model each agent is on makes that visible in a glance
#instead of after a month.
@router.get("/stats")
async def get_stats(
    _claims: Annotated[dict[str, Any], Depends(require_auth)],
) -> StatsResponse:
    settings = get_settings()
    numbers = await cache.stats()

    routing = []
    for role in AgentRole:
        model = model_for(role)
        price = price_for(model)
        routing.append(
            RoleRouting(
                role=role.value,
                model=model,
                input_per_million_usd=price.input_per_million,
                output_per_million_usd=price.output_per_million,
                cacheable=role.value in settings.semantic_cache_roles,
            )
        )

    return StatsResponse(
        cache=CacheStats(
            enabled=settings.semantic_cache_enabled,
            threshold=settings.semantic_cache_threshold,
            hits=int(numbers["hits"]),
            misses=int(numbers["misses"]),
            lookups=int(numbers["lookups"]),
            hit_rate=float(numbers["hit_rate"]),
            tokens_saved=int(numbers["tokens_saved"]),
        ),
        routing=routing,
    )
