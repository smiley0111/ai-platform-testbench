"""限流与熔断。"""

from __future__ import annotations

import math
import threading
import time


class TokenBucket:
    """每个 API Key 一个桶：容量 = rpm，按 rpm/60 每秒回填。"""

    def __init__(self, rpm: int):
        self.capacity = float(rpm)
        self.tokens = float(rpm)
        self.rate = rpm / 60.0
        self.updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """拿到令牌返回 0；拿不到返回需要等待的秒数。"""
        with self._lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
            self.updated = now
            if self.tokens >= 1:
                self.tokens -= 1
                return 0.0
            return (1 - self.tokens) / self.rate


def retry_after_header(wait: float) -> str:
    return str(max(1, math.ceil(wait)))


class CircuitBreaker:
    """连续失败达到阈值后打开；打开 reset 秒后进入半开，只放一个试探请求。"""

    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, failures: int, reset: float):
        self.threshold = failures
        self.reset = reset
        self.state = self.CLOSED
        self.failures = 0
        self.opened_at = 0.0
        self._probe_in_flight = False
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self.state == self.CLOSED:
                return True
            if self.state == self.OPEN and time.monotonic() - self.opened_at >= self.reset:
                self.state = self.HALF_OPEN
                self._probe_in_flight = False
            if self.state == self.HALF_OPEN and not self._probe_in_flight:
                self._probe_in_flight = True
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self.state = self.CLOSED
            self.failures = 0
            self._probe_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            self._probe_in_flight = False
            if self.state == self.HALF_OPEN:
                self._open()
                return
            self.failures += 1
            if self.failures >= self.threshold:
                self._open()

    def _open(self) -> None:
        self.state = self.OPEN
        self.opened_at = time.monotonic()
        self.failures = 0
