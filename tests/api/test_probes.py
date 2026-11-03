"""部署相关的接口：存活与就绪检查、版本和实例响应头、从环境变量读配置。K8s 用例依赖这些行为。"""

from __future__ import annotations

import socket
import time

from fastapi.testclient import TestClient

from testbench.gateway import GatewayConfig, create_app


def client(**kw) -> TestClient:
    return TestClient(create_app(GatewayConfig(**kw)))


def test_ready_after_startup_delay():
    c = client(startup_delay=0.5)
    assert c.get("/readyz").status_code == 503
    assert c.get("/healthz").status_code == 200, "预热期间存活检查必须通过，否则 K8s 会反复重启"
    time.sleep(0.6)
    assert c.get("/readyz").status_code == 200


def test_broken_release_never_ready_but_alive():
    c = client(fail_ready=True)
    assert c.get("/readyz").status_code == 503
    assert c.get("/healthz").status_code == 200


def test_version_and_instance_headers_on_every_response():
    c = client(version="v2")
    for path in ("/healthz", "/readyz", "/v1/models"):
        r = c.get(path)
        assert r.headers["x-gateway-version"] == "v2"
        assert r.headers["x-served-by"] == socket.gethostname()
    assert c.get("/v1/models").status_code == 401, "错误响应也要带版本头，灰度时才能统计新版本的错误率"


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("TESTBENCH_VERSION", "v3")
    monkeypatch.setenv("TESTBENCH_STARTUP_DELAY", "2.5")
    monkeypatch.setenv("TESTBENCH_FAIL_READY", "1")
    monkeypatch.setenv("TESTBENCH_EXTRA_KEYS", "sk-a:6,sk-b:120")
    cfg = GatewayConfig.default()
    assert (cfg.version, cfg.startup_delay, cfg.fail_ready) == ("v3", 2.5, True)
    assert cfg.keys["sk-a"].rpm == 6 and cfg.keys["sk-b"].rpm == 120
    assert "sk-local-dev" in cfg.keys
