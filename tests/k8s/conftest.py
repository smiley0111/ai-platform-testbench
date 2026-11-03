"""K8s 用例只在设置 TESTBENCH_K8S=1 时运行（CI 里的 kind 集群），避免误操作本机 kubectl 指向的真实集群。

每个用例开始前把集群恢复到基线：稳定版 v1 两个副本、没有灰度版、没有 HPA。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from kube import NS, apply, gateway, kubectl, pods, ready_pods, rollout, send, wait_for

REPORT = Path(__file__).resolve().parents[2] / "reports" / "k8s" / "results.json"


def pytest_collection_modifyitems(config, items):
    if os.getenv("TESTBENCH_K8S") == "1":
        return
    skip = pytest.mark.skip(reason="需要 kind 集群，设置 TESTBENCH_K8S=1 后运行")
    for item in items:
        if "k8s" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def results():
    """用例把实测数据写进来，结束后存成 JSON，CI 把它贴到运行摘要里。"""
    data: dict = {}
    yield data
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.fixture(scope="session")
def cluster():
    apply([{"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": NS}}])


@pytest.fixture
def baseline(cluster):
    kubectl("delete", "hpa", "gateway", "--ignore-not-found")
    kubectl("delete", "deployment", "gateway-canary", "--ignore-not-found")
    apply(gateway())
    rollout("gateway")
    wait_for(lambda: len(pods()) == 2 and len(ready_pods()) == 2, 90, "基线：只剩 2 个就绪的稳定版 Pod")
    wait_for(lambda: send().status == 200, 60, "NodePort 可访问")
