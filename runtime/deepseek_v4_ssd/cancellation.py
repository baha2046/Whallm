"""Cooperative cancellation scoped to the synchronous request thread.

The socket observer only sets an Event. MLX work and generator cleanup stay on
the generation thread; CLI/research callers outside this scope are unaffected.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event


class GenerationCancelled(Exception):
    """The client no longer needs this generation."""


_cancellation: ContextVar[Event | None] = ContextVar("generation_cancellation", default=None)


def check_cancelled() -> None:
    event = _cancellation.get()
    if event is not None and event.is_set():
        raise GenerationCancelled("Client disconnected")


@contextmanager
def cancellation_scope(event: Event):
    token = _cancellation.set(event)
    try:
        yield
    finally:
        _cancellation.reset(token)
