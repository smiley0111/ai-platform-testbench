"""弹性伸缩：HPA 按 CPU 扩容、负载撤掉后缩容且不丢请求；扩容后单实例限流被放大。"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from kube import (LIMITED_KEY, LIMITED_RPM, Load, apply, gateway, hpa, kjson, pods, ready_pods, send,
                  wait_for)

pytestmark = pytest.mark.k8s


def _hpa_status() -> dict:
    return kjson("get", "hpa", "gateway").get("status", {})


def _cpu_utilization() -> int | None:
    for m in _hpa_status().get("currentMetrics") or []:
        cur = m.get("resource", {}).get("current", {})
        if "averageUtilization" in cur:
            return cur["averageUtilization"]
    return None


def test_hpa_scales_out_under_load_and_back_in(baseline, results):
    apply(hpa())
    wait_for(lambda: _cpu_utilization() is not None, 180, "HPA 拿到 CPU 指标（metrics-server 采集周期约 15 秒）",
             interval=5)
    idle = _cpu_utilization()

    start = time.monotonic()
    with Load(24) as heavy:
        wait_for(lambda: len(ready_pods()) > 2, 240, "负载下扩容出新的就绪副本", interval=3)
        scaled_out = time.monotonic() - start
        peak_cpu = _cpu_utilization()
        time.sleep(10)
        served = {r.pod for r in heavy.since(time.monotonic() - 10)}
    peak = len(ready_pods())
    assert 2 < peak <= 5
    assert len(served) > 2, "新副本应该接到流量"
    assert not heavy.failures(), heavy.summary()

    start = time.monotonic()
    with Load(1) as light:  # 缩容期间保持少量请求，检查删 Pod 时不丢请求
        wait_for(lambda: len(pods()) == 2, 300, "负载撤掉后缩回最小副本数", interval=5)
        scaled_in = time.monotonic() - start
    assert not light.failures(), light.summary()

    results["hpa"] = {"idle_cpu_pct": idle, "peak_cpu_pct": peak_cpu, "peak_replicas": peak,
                      "scale_out_s": round(scaled_out), "scale_in_s": round(scaled_in),
                      "under_load": heavy.summary(), "during_scale_in": light.summary()}


def test_rate_limit_is_per_replica_not_global(baseline, results):
    """令牌桶在每个进程的内存里，N 个副本时同一个 Key 实际能用 N 倍配额，扩容或重启 Pod 还会重置配额。
    这个用例刻画的是现状（已知缺陷）：改成 Redis 等共享存储的全局限流后，应改为断言放行数不超过单实例配额。"""
    replicas = 3
    apply(gateway(replicas=replicas))
    wait_for(lambda: len(ready_pods()) == replicas, 90, f"{replicas} 个副本就绪")

    with ThreadPoolExecutor(12) as pool:
        recs = list(pool.map(lambda _: send(LIMITED_KEY), range(90)))
    admitted = [r for r in recs if r.status == 200]
    limited = [r for r in recs if r.status == 429]
    assert len(admitted) + len(limited) == len(recs)
    assert len(admitted) > LIMITED_RPM, "多副本下放行数超过了单个 Key 的配额"
    assert len(admitted) <= replicas * LIMITED_RPM + replicas, "每个副本最多放行一桶（加少量回填）"
    assert len({r.pod for r in admitted}) == replicas
    results["rate_limit"] = {"replicas": replicas, "rpm_per_key": LIMITED_RPM,
                             "requests": len(recs), "admitted": len(admitted)}
