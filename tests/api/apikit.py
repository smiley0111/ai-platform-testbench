"""API 用例的常量和工具函数。放在普通模块里，避免和其他目录的 conftest 重名冲突。"""

from __future__ import annotations

import json
import socket
import threading
import time

import httpx
import uvicorn

from testbench.gateway import GatewayConfig, KeyPolicy, ModelSpec, create_app

ADMIN_TOKEN = "admin-secret"
KEY_ALL = "sk-test-all"
KEY_LIMITED = "sk-test-limited"  # 只能用 qwen-mini
KEY_RATE = "sk-test-ratelimit"  # 每分钟 3 次
TIMEOUT = 0.5
BREAKER_FAILURES = 3
BREAKER_RESET = 0.5
MAX_RETRIES = 2


def make_config() -> GatewayConfig:
    return GatewayConfig(
        backend="mock",
        keys={
            KEY_ALL: KeyPolicy(rpm=100_000),
            KEY_LIMITED: KeyPolicy(models={"qwen-mini"}, rpm=100_000),
            KEY_RATE: KeyPolicy(rpm=3),
        },
        models={
            "qwen-mini": ModelSpec("mock-chat", fallback="qwen-backup"),
            "qwen-backup": ModelSpec("mock-chat"),
            "no-fallback": ModelSpec("mock-chat"),
            "qwen-vl": ModelSpec("mock-vl", vision=True),
            "embed-mini": ModelSpec("mock-embed", kind="embedding"),
        },
        request_timeout=TIMEOUT,
        max_retries=MAX_RETRIES,
        retry_backoff=0.01,
        breaker_failures=BREAKER_FAILURES,
        breaker_reset=BREAKER_RESET,
        blocked_words=["违禁词示例"],
        admin_token=ADMIN_TOKEN,
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ThreadServer(uvicorn.Server):
    def install_signal_handlers(self) -> None:  # 非主线程不能注册信号
        pass


def start_server(app) -> tuple[str, _ThreadServer, threading.Thread]:
    port = _free_port()
    server = _ThreadServer(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("gateway did not start")
        time.sleep(0.02)
    return f"http://127.0.0.1:{port}", server, thread


def http_client(base_url: str, **kwargs) -> httpx.Client:
    """trust_env=False：不读系统代理设置。Windows 上每次读注册表要约 0.6 秒，本地测试也不需要代理。"""
    return httpx.Client(base_url=base_url, timeout=10, trust_env=False, **kwargs)


class Admin:
    def __init__(self, base_url: str):
        self.http = http_client(base_url, headers={"x-admin-token": ADMIN_TOKEN})

    def fault(self, model: str = "*", **kwargs) -> None:
        self.http.post("/admin/faults", json={"model": model, **kwargs}).raise_for_status()

    def reset(self) -> None:
        self.http.post("/admin/reset").raise_for_status()

    def state(self) -> dict:
        return self.http.get("/admin/state").json()


def chat_body(text: str = "hello world", model: str = "qwen-mini", **kwargs) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": text}], **kwargs}


def assert_error(resp: httpx.Response, status: int, code: str | None = None,
                 param: str | None = None) -> dict:
    """所有错误都必须是 OpenAI 格式：{"error": {"message", "type", "param", "code"}}。"""
    assert resp.status_code == status, resp.text
    err = resp.json()["error"]
    assert set(err) == {"message", "type", "param", "code"}
    assert err["message"] and err["type"]
    if code is not None:
        assert err["code"] == code
    if param is not None:
        assert err["param"] is not None and param in err["param"]
    return err


def read_sse(resp: httpx.Response) -> list[str]:
    """按 SSE 规范读出所有 data 字段（不含 'data: ' 前缀）。"""
    events = []
    for line in resp.iter_lines():
        if line.startswith("data: "):
            events.append(line[len("data: "):])
        else:
            assert line == "", f"unexpected SSE line: {line!r}"
    return events


def sse_json(events: list[str]) -> list[dict]:
    return [json.loads(e) for e in events if e != "[DONE]"]
