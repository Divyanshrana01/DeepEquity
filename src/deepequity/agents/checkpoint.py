from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from deepequity.core.config import get_settings
from deepequity.core.logging import get_logger

logger = get_logger("deepequity.agents.checkpoint")

_saver: BaseCheckpointSaver | None = None
#which backend we ended up on. worth exposing rather than only logging: the fallback is
#deliberately quiet so a run still completes, but "checkpointing is off" is exactly the
#sort of degradation that otherwise goes unnoticed until someone needs a resume and finds
#there was never anything saved.
_backend: str = "not initialised"


def checkpoint_backend() -> str:
    return _backend


#gives the graph somewhere to save its state after every node.
#
#the point is resumability. a research run takes minutes and costs real tokens, so if the
#api restarts halfway through, starting again from the ticker means paying for the
#planning, the retrieval and the first debate round a second time. with a checkpoint the
#run picks up from the last node that finished.
#
#it also makes a run inspectable while it's still going, which is how the polling endpoint
#can report which stage it's on rather than just "still working".
async def get_checkpointer() -> BaseCheckpointSaver:
    global _saver, _backend
    if _saver is not None:
        return _saver

    settings = get_settings()
    try:
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver

        saver = AsyncRedisSaver(redis_url=settings.redis_url)
        await saver.asetup()
        _saver = saver
        _backend = "redis"
        logger.info("checkpointer_ready", backend="redis")
    except Exception as exc:  # noqa: BLE001 - any failure here should degrade, not crash
        #falling back rather than failing is deliberate. losing resumability makes runs
        #more expensive after a restart, it doesn't make them wrong, and refusing to do
        #any research at all because redis checkpointing wouldn't initialise would be a
        #much worse outcome than quietly losing a feature.
        logger.warning("checkpointer_fallback_to_memory", error=str(exc))
        _saver = InMemorySaver()
        _backend = f"in-memory (redis unavailable: {str(exc)[:120]})"

    return _saver


async def reset_checkpointer() -> None:
    global _saver, _backend
    _saver = None
    _backend = "not initialised"
