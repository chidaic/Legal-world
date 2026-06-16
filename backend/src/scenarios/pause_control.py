"""Shared pause controller helpers for local single-case debug runs."""

from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from typing import Iterator


class RuntimePauseEvent:
    """Thread-friendly pause signal that supports both sync and async waits."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._event.set()

    def set(self) -> None:
        self._event.set()

    def clear(self) -> None:
        self._event.clear()

    async def wait(self) -> None:
        await asyncio.to_thread(self._event.wait)

    def wait_sync(self) -> None:
        self._event.wait()


class RuntimePauseController:
    """Minimal controller exposing the same fields expected by BaseScenario."""

    def __init__(self) -> None:
        self._paused = False
        self._resumed_event = RuntimePauseEvent()

    def pause(self) -> None:
        self._paused = True
        self._resumed_event.clear()

    def resume(self) -> None:
        self._paused = False
        self._resumed_event.set()


_RUNTIME_PAUSE_LOCAL = threading.local()


def get_runtime_pause_controller() -> RuntimePauseController | None:
    controller = getattr(_RUNTIME_PAUSE_LOCAL, "controller", None)
    return controller if isinstance(controller, RuntimePauseController) else None


@contextmanager
def runtime_pause_controller_context(controller: RuntimePauseController | None) -> Iterator[None]:
    previous = getattr(_RUNTIME_PAUSE_LOCAL, "controller", None)
    _RUNTIME_PAUSE_LOCAL.controller = controller
    try:
        yield
    finally:
        _RUNTIME_PAUSE_LOCAL.controller = previous
