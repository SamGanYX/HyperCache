"""
Single-Flight Incast Shield and Request Coalescing.

Prevents "thundering herd" / incast storms in distributed KV-cache architectures
when multiple concurrent queries or worker threads request the same missing KV block
simultaneously over TCP, RDMA, or from disk storage.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Callable, Generic, Optional, TypeVar

T = TypeVar("T")


class _FlightCall(Generic[T]):
    __slots__ = ("event", "val", "err", "waiters")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.val: Optional[T] = None
        self.err: Optional[Exception] = None
        self.waiters: int = 0


class SingleFlight(Generic[T]):
    """
    Suppresses duplicate concurrent function executions for identical keys.
    If multiple threads call `do(key, fn)` simultaneously with the same key,
    only the first thread (the leader) invokes `fn`. All other callers wait
    for the leader's execution and receive the identical result.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: dict[str, _FlightCall[T]] = {}
        self._total_deduplicated = 0
        self._total_calls = 0

    @property
    def total_deduplicated(self) -> int:
        with self._lock:
            return self._total_deduplicated

    @property
    def total_calls(self) -> int:
        with self._lock:
            return self._total_calls

    def do(self, key: str, fn: Callable[[], T]) -> tuple[T, bool]:
        """
        Executes fn for key or waits for an ongoing execution of key.
        Returns:
            (result, is_leader): is_leader is True if this invocation executed fn,
            and False if it was deduplicated and received the shared result.
        """
        with self._lock:
            self._total_calls += 1
            if key in self._calls:
                call = self._calls[key]
                call.waiters += 1
                self._total_deduplicated += 1
                is_leader = False
            else:
                call = _FlightCall[T]()
                self._calls[key] = call
                is_leader = True

        if not is_leader:
            call.event.wait()
            if call.err is not None:
                raise call.err
            assert call.val is not None
            return call.val, False

        try:
            val = fn()
            call.val = val
            return val, True
        except Exception as exc:
            call.err = exc
            raise
        finally:
            with self._lock:
                self._calls.pop(key, None)
            call.event.set()
