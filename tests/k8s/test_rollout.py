"""发布：滚动更新不丢请求、坏版本被挡住并回滚、灰度按副本比例分流后撤回。"""

from __future__ import annotations

import math
import time

import pytest
from kube import Load, apply, canary, gateway, kubectl, pods, ready_pods, rollout, wait_for

pytestmark = pytest.mark.k8s


def _rolling_update(prestop: bool) -> Load:
    with Load(8) as load:
        time.sleep(2)
        apply(gateway(version="v2", prestop=prestop))
        rollout("gateway")
        wait_for(lambda: len(pods()) == 2, 90, "旧 Pod 全部退出")
        time.sleep(3)
    return load


def test_rolling_update_without_failed_requests(baseline, results):
    load = _rolling_update(prestop=True)
    versions = [r.version for r in load.records]
    assert "v1" in versions and "v2" in versions
    assert not load.failures(), load.summary()
    assert {r.version for r in load.records[-20:]} == {"v2"}, "更新完成后只应该由新版本响应"
    results["rolling_update"] = load.summary()


def test_rolling_update_without_prestop(baseline, results):
    """对照组：去掉 preStop 后再滚动一次。摘除 Endpoint 和发 SIGTERM 是并行的，
    进程先停止监听时，还在路由规则里的 Pod 会拒绝新连接。失败数取决于时序，只记录不断言。"""
    apply(gateway(prestop=False))
    rollout("gateway")
    wait_for(lambda: len(pods()) == 2, 90, "无 preStop 的 v1 就位")
    load = _rolling_update(prestop=False)
    results["rolling_update_without_prestop"] = load.summary()


def test_broken_release_is_held_back_and_rolled_back(baseline, results):
    """新版本就绪检查一直失败：maxUnavailable=0 保证旧版本一个都不下线，流量全在旧版本。"""
    old = set(ready_pods())
    with Load(4) as load:
        apply(gateway(version="v2-broken", env={"TESTBENCH_FAIL_READY": "1"}))
        wait_for(lambda: len(pods()) == 3, 60, "新版本 Pod 已创建")
        time.sleep(15)
        assert set(ready_pods()) == old, "坏版本不应该替换掉任何旧 Pod"
        kubectl("rollout", "undo", "deployment/gateway")
        rollout("gateway")
        wait_for(lambda: len(pods()) == 2, 90, "坏版本 Pod 被删除")
    assert {r.version for r in load.records} == {"v1"}
    assert not load.failures(), load.summary()
    assert set(ready_pods()) == old, "回滚不应该重建旧 Pod"
    results["broken_release"] = load.summary()


def test_canary_traffic_matches_replica_ratio_and_rolls_back(baseline, results):
    apply(gateway(replicas=3))
    apply(canary(replicas=1))
    wait_for(lambda: len(ready_pods("stable")) == 3 and len(ready_pods("canary")) == 1, 90, "3 稳定 + 1 灰度")

    with Load(8) as load:
        wait_for(lambda: len(load.records) >= 600, 120, "采够 600 个样本")
    n = len(load.records)
    share = sum(r.version == "v2" for r in load.records) / n
    expected = 1 / 4
    tolerance = 4 * math.sqrt(expected * (1 - expected) / n)  # 二项分布 4 个标准差
    assert abs(share - expected) < tolerance, f"灰度流量 {share:.1%}，期望 {expected:.0%} ± {tolerance:.1%}"
    assert not load.failures()

    with Load(8) as rollback:
        time.sleep(2)
        kubectl("delete", "deployment", "gateway-canary", "--wait=false")
        wait_for(lambda: not pods("canary"), 90, "灰度 Pod 全部退出")
        done = time.monotonic()
        time.sleep(3)
    assert {r.version for r in rollback.since(done)} == {"v1"}
    assert not rollback.failures(), rollback.summary()
    results["canary"] = {"samples": n, "canary_share": round(share, 4), "expected": expected,
                         "tolerance": round(tolerance, 4), "rollback": rollback.summary()}
