"""Small dependency-free tracing context for logs and diagnostics."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator
from uuid import uuid4


trace_id_context: ContextVar[str | None] = ContextVar("trace_id", default=None)


def current_trace_id() -> str | None:
    return trace_id_context.get()


@contextmanager
def trace(trace_id: str | None = None) -> Iterator[str]:
    value = trace_id or str(uuid4())
    token = trace_id_context.set(value)
    try:
        yield value
    finally:
        trace_id_context.reset(token)
