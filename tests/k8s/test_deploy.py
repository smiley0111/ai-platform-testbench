"""部署与自愈：负载均衡、就绪检查挡流量、删 Pod 不中断服务、进程退出自动拉起。"""

from __future__ import annotations

import time

import pytest
from kube import Load, apply, canary, kubectl, pods, ready_pods, restart_count, send, wait_for

pytestmark = pytest.mark.k8s


def test_service_balances_across_ready_replicas(baseline):
    recs = [send() for _ in range(40)]
    assert all(r.status == 200 for r in recs)
    assert {r.pod for r in recs} == set(ready_pods()), "每个副本都应该接到流量"
    assert {r.version for r in recs} == {"v1"}


def test_unready_pod_receives_no_traffic(baseline, results):
    """新 Pod 预热（加载模型）期间进程已经在跑，但就绪检查不通过，不能接流量。"""
    warmup = 15
    apply(canary(env={"TESTBENCH_STARTUP_DELAY": str(warmup)}))
    pod = wait_for(lambda: [p for p in pods("canary") if p["status"].get("phase") == "Running"],
                   60, "灰度 Pod 进入 Running")[0]["metadata"]["name"]
    with Load(4) as load:
        wait_for(lambda: pod in ready_pods("canary"), warmup + 30, "灰度 Pod 就绪")
        became_ready = time.monotonic()
        wait_for(lambda: any(r.version == "v2" for r in load.since(became_ready)), 30, "就绪后开始接流量")
    before = [r for r in load.records if r.t < became_ready - 2]  # 留 2 秒余量给探针周期和规则同步
    assert before, "预热期间应该有请求"
    assert all(r.version == "v1" for r in before), "未就绪的 Pod 不应该接到请求"
    assert not load.failures()
    results["readiness_gate"] = {"warmup_s": warmup, "requests_while_warming": len(before), **load.summary()}


def test_deleted_pod_is_replaced_without_failed_requests(baseline, results):
    victim = ready_pods()[0]
    with Load(8) as load:
        time.sleep(2)
        kubectl("delete", "pod", victim, "--wait=false")
        wait_for(lambda: victim not in ready_pods() and len(ready_pods()) == 2, 90, "替补 Pod 就绪")
        time.sleep(2)
    assert victim not in {p["metadata"]["name"] for p in pods()}
    assert not load.failures(), load.summary()
    results["pod_delete"] = load.summary()


def test_exited_process_is_restarted_in_place(baseline, results):
    """进程自己退出（崩溃）时，kubelet 在同一个 Pod 里重启容器，Pod 名不变。"""
    pod = ready_pods()[0]
    before = restart_count(pod)
    with Load(4) as load:
        time.sleep(1)
        kubectl("exec", pod, "--", "python", "-c", "import os, signal; os.kill(1, signal.SIGTERM)",
                check=False)  # 容器退出时 exec 会话可能被一起断开
        wait_for(lambda: restart_count(pod) == before + 1, 60, "容器重启计数加 1")
        wait_for(lambda: pod in ready_pods(), 60, "重启后重新就绪")
        time.sleep(1)
    assert send().status == 200
    # 进程直接退出时没有 preStop 缓冲，退出到探针发现之间转过来的请求会失败。只记录不断言。
    results["process_exit"] = load.summary()
