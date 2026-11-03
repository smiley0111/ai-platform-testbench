"""在后台线程里起一个真实的网关 HTTP 服务（Mock 后端），测试全部走真实网络请求。"""

from __future__ import annotations

import pytest
from apikit import KEY_ALL, Admin, http_client, make_config, start_server

from testbench.gateway import create_app


@pytest.fixture(scope="session")
def base_url():
    url, server, thread = start_server(create_app(make_config()))
    yield url
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="session")
def admin(base_url) -> Admin:
    return Admin(base_url)


@pytest.fixture(autouse=True)
def _clean_state(admin):
    """每个用例前清空故障、熔断器和限流桶，保证用例之间互不影响。"""
    admin.reset()


@pytest.fixture(scope="session")
def client(base_url):
    """默认带全权限 key 的客户端；单个请求可以用 headers= 覆盖。"""
    with http_client(base_url, headers={"authorization": f"Bearer {KEY_ALL}"}) as c:
        yield c


@pytest.fixture(scope="session")
def anon(base_url):
    """不带任何鉴权头的客户端，用来测鉴权。"""
    with http_client(base_url) as c:
        yield c
