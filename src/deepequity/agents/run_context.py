from __future__ import annotations

from contextvars import ContextVar

#which run the current bit of work belongs to.
#
#a context variable rather than an argument threaded through every agent, because the only
#thing that needs it is the cache, several layers below, and adding a run_id parameter to
#four agents and the llm client to serve one caller would be a lot of noise for it. async
#tasks inherit the context they were created in, so the debate agents running under a task
#group see the same value without being told.
_current_run: ContextVar[str | None] = ContextVar("current_run_id", default=None)


def set_current_run(run_id: str) -> None:
    _current_run.set(run_id)


def current_run_id() -> str | None:
    return _current_run.get()
