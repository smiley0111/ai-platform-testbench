"""可靠性：重试、超时、熔断（打开 / 半开 / 恢复）、降级到备用模型、并发。

全部用 Mock 后端注入故障，保证每次运行结果一致。
"""

import time
from concurrent.futures import ThreadPoolExecutor

from conftest import (
    BREAKER_FAILURES,
    BREAKER_RESET,
    MAX_RETRIES,
    TIMEOUT,
    assert_error,
    chat_body,
)


def call(client, model="no-fallback"):
    return client.post("/v1/chat/completions", json=chat_body(model=model))


def open_breaker(client, admin, model="no-fallback"):
    """有备用模型时请求会被降级成功，所以这里只看熔断器状态，不看返回码。"""
    admin.fault(model, mode="error", status=500)
    for _ in range(BREAKER_FAILURES):
        call(client, model)
    assert admin.state()["breakers"][model] == "open"


class TestRetry:
    def test_transient_errors_are_retried(self, client, admin):
        admin.fault("no-fallback", mode="flaky", fail_times=MAX_RETRIES)
        assert call(client).status_code == 200
        assert admin.state()["calls"]["no-fallback"] == MAX_RETRIES + 1

    def test_gives_up_after_max_retries(self, client, admin):
        admin.fault("no-fallback", mode="flaky", fail_times=MAX_RETRIES + 1)
        assert_error(call(client), 502, "upstream_error")
        assert admin.state()["calls"]["no-fallback"] == MAX_RETRIES + 1

    def test_client_errors_are_not_retried(self, client, admin):
        """后端返回 4xx 说明请求本身有问题，重试只会浪费资源。"""
        admin.fault("no-fallback", mode="error", status=400)
        assert_error(call(client), 400, "upstream_rejected")
        assert admin.state()["calls"]["no-fallback"] == 1

    def test_backend_429_is_retried(self, client, admin):
        admin.fault("no-fallback", mode="flaky", status=429, fail_times=1)
        assert call(client).status_code == 200


class TestTimeout:
    def test_slow_backend_returns_504_on_time(self, client, admin):
        admin.fault("no-fallback", mode="slow", delay=TIMEOUT * 3)
        start = time.monotonic()
        assert_error(call(client), 504, "upstream_timeout")
        assert time.monotonic() - start < TIMEOUT * 2, "超时要按配置及时返回，不能等后端"

    def test_slightly_slow_backend_still_succeeds(self, client, admin):
        admin.fault("no-fallback", mode="slow", delay=TIMEOUT / 5)
        assert call(client).status_code == 200


class TestCircuitBreaker:
    def test_opens_after_consecutive_failures_and_fails_fast(self, client, admin):
        open_breaker(client, admin)
        calls_before = admin.state()["calls"]["no-fallback"]
        start = time.monotonic()
        assert_error(call(client), 503, "circuit_open")
        assert time.monotonic() - start < 0.2, "熔断打开后应快速失败"
        assert admin.state()["calls"]["no-fallback"] == calls_before, "熔断打开后不应再请求后端"

    def test_success_resets_failure_count(self, client, admin):
        admin.fault("no-fallback", mode="error", status=500)
        for _ in range(BREAKER_FAILURES - 1):
            call(client)
        admin.fault("no-fallback", mode="ok")
        assert call(client).status_code == 200
        admin.fault("no-fallback", mode="error", status=500)
        call(client)
        assert admin.state()["breakers"]["no-fallback"] == "closed"

    def test_half_open_probe_success_closes(self, client, admin):
        open_breaker(client, admin)
        admin.fault("no-fallback", mode="ok")
        time.sleep(BREAKER_RESET + 0.1)
        assert call(client).status_code == 200
        assert admin.state()["breakers"]["no-fallback"] == "closed"

    def test_half_open_probe_failure_reopens(self, client, admin):
        open_breaker(client, admin)
        time.sleep(BREAKER_RESET + 0.1)
        assert_error(call(client), 502)
        assert admin.state()["breakers"]["no-fallback"] == "open"
        assert_error(call(client), 503, "circuit_open")

    def test_breakers_are_isolated_per_model(self, client, admin):
        open_breaker(client, admin)
        assert call(client, "qwen-backup").status_code == 200


class TestFallback:
    def test_falls_back_to_backup_model(self, client, admin):
        admin.fault("qwen-mini", mode="error", status=500)
        r = call(client, "qwen-mini")
        assert r.status_code == 200
        assert r.headers["x-fallback"] == "true"
        assert r.headers["x-served-model"] == "qwen-backup"
        assert r.json()["model"] == "qwen-backup", "响应里要如实告诉用户实际是哪个模型回答的"

    def test_normal_request_is_not_marked_fallback(self, client):
        r = call(client, "qwen-mini")
        assert r.headers["x-fallback"] == "false"

    def test_primary_breaker_open_goes_straight_to_backup(self, client, admin):
        open_breaker(client, admin, "qwen-mini")
        primary_calls = admin.state()["calls"]["qwen-mini"]
        r = call(client, "qwen-mini")
        assert r.status_code == 200 and r.headers["x-served-model"] == "qwen-backup"
        assert admin.state()["calls"]["qwen-mini"] == primary_calls

    def test_both_down_returns_error(self, client, admin):
        admin.fault("*", mode="error", status=500)
        assert_error(call(client, "qwen-mini"), 502, "upstream_error")

    def test_streaming_falls_back_too(self, client, admin):
        admin.fault("qwen-mini", mode="error", status=500)
        with client.stream("POST", "/v1/chat/completions",
                           json=chat_body(model="qwen-mini", stream=True)) as r:
            assert r.status_code == 200
            assert r.headers["x-served-model"] == "qwen-backup"


class TestConcurrency:
    def test_parallel_requests_all_succeed(self, client):
        with ThreadPoolExecutor(max_workers=20) as pool:
            results = list(pool.map(lambda i: call(client, "qwen-mini").status_code, range(50)))
        assert results == [200] * 50
