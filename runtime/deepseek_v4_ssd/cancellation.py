"""Cooperative cancellation scoped to the synchronous request thread.

The socket observer only sets an Event. MLX work and generator cleanup stay on
the generation thread; CLI/research callers outside this scope are unaffected.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from concurrent.futures import FIRST_EXCEPTION, wait
from threading import Event


class GenerationCancelled(Exception):
    """The client no longer needs this generation."""


_cancellation: ContextVar[Event | None] = ContextVar("generation_cancellation", default=None)


def check_cancelled() -> None:
    event = _cancellation.get()
    if event is not None and event.is_set():
        raise GenerationCancelled("Client disconnected")


def cancel_and_drain(futures) -> None:
    """Stop queued reads and retain their buffers until running writes finish."""
    futures = tuple(futures)
    for future in futures:
        future.cancel()
    for future in futures:
        if not future.cancelled():
            future.exception()


def wait_for_futures(futures) -> tuple:
    """Wait on request-owned reads with cancellation; always drain before raising."""
    futures = tuple(futures)
    try:
        check_cancelled()
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=0.05, return_when=FIRST_EXCEPTION)
            check_cancelled()
            for future in done:
                future.result()
        return tuple(future.result() for future in futures)
    except BaseException:
        cancel_and_drain(futures)
        raise


@contextmanager
def cancellation_scope(event: Event):
    token = _cancellation.set(event)
    try:
        yield
    finally:
        _cancellation.reset(token)
