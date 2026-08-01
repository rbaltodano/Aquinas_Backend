"""Priority-aware serialization for the process-scoped MLX model."""

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Condition, Event
import time
from collections.abc import Iterator


class GenerationPreempted(RuntimeError):
    """Raised when foreground work interrupts a background generation."""


@dataclass(frozen=True)
class GenerationLease:
    cancel_event: Event
    queue_wait_seconds: float


class GenerationCoordinator:
    """Serialize generation and signal active background work to yield."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._active_token: object | None = None
        self._active_priority: str | None = None
        self._active_cancel_event: Event | None = None
        self._foreground_waiters = 0

    @contextmanager
    def session(self, priority: str) -> Iterator[GenerationLease]:
        if priority not in {"foreground", "background"}:
            raise ValueError(f"Unsupported generation priority: {priority}")

        requested_at = time.perf_counter()
        token = object()
        with self._condition:
            if priority == "foreground":
                self._foreground_waiters += 1
                if (
                    self._active_priority == "background"
                    and self._active_cancel_event is not None
                ):
                    self._active_cancel_event.set()

            try:
                while (
                    self._active_token is not None
                    or (priority == "background" and self._foreground_waiters > 0)
                ):
                    self._condition.wait()
                self._active_token = token
                self._active_priority = priority
                self._active_cancel_event = Event()
            finally:
                if priority == "foreground":
                    self._foreground_waiters -= 1

            lease = GenerationLease(
                cancel_event=self._active_cancel_event,
                queue_wait_seconds=time.perf_counter() - requested_at,
            )

        try:
            yield lease
        finally:
            with self._condition:
                if self._active_token is token:
                    self._active_token = None
                    self._active_priority = None
                    self._active_cancel_event = None
                    self._condition.notify_all()
