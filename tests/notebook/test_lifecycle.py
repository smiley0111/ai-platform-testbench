"""内核生命周期与多用户：重启、关闭与资源清理、配额、会话隔离、并发、空闲回收。"""

from __future__ import annotations

import threading
import time

import psutil
import pytest

from testbench.notebook import NotebookService, QuotaExceeded, SessionNotFound

pytestmark = pytest.mark.notebook


def test_restart_clears_state_and_replaces_process(svc, fresh):
    svc.execute(fresh, "secret = 1")
    old_pid = svc.pid(fresh)
    svc.restart(fresh)
    assert svc.pid(fresh) != old_pid
    assert not psutil.pid_exists(old_pid) or psutil.Process(old_pid).status() == psutil.STATUS_ZOMBIE
    r = svc.execute(fresh, "secret")
    assert r.error["ename"] == "NameError"
    assert svc.execute(fresh, "1").execution_count == 2  # 计数从头开始：NameError 那次是 1


def test_shutdown_releases_process_and_connection_file(svc):
    sid = svc.start("leaver")
    pid, conn = svc.pid(sid), svc.connection_file(sid)
    assert conn.exists()
    svc.shutdown(sid)
    deadline = time.monotonic() + 5
    while psutil.pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not psutil.pid_exists(pid)
    assert not conn.exists()  # 连接文件里有密钥，不能留在磁盘上
    assert sid not in svc.sessions()


def test_using_closed_session_is_rejected(svc):
    sid = svc.start("closer")
    svc.shutdown(sid)
    with pytest.raises(SessionNotFound):
        svc.execute(sid, "1")
    with pytest.raises(SessionNotFound):
        svc.shutdown(sid)


def test_sessions_are_isolated(svc, fresh):
    other = svc.start("someone-else")
    try:
        svc.execute(fresh, "private = 'mine'")
        r = svc.execute(other, "private")
        assert r.error["ename"] == "NameError"
        assert svc.pid(fresh) != svc.pid(other)
    finally:
        svc.shutdown(other)


def test_different_kernels_run_in_parallel(svc, fresh):
    other = svc.start("parallel")
    try:
        results = {}

        def run(sid):
            results[sid] = svc.execute(sid, "import time; time.sleep(1.5)")

        t0 = time.monotonic()
        threads = [threading.Thread(target=run, args=(s,)) for s in (fresh, other)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert time.monotonic() - t0 < 2.8
        assert all(r.status == "ok" for r in results.values())
    finally:
        svc.shutdown(other)


def test_same_kernel_requests_are_queued(svc, fresh):
    """同一个内核同时收到两段代码要排队执行，不能交错，也不能丢请求。"""
    results = []

    def run(tag):
        results.append(svc.execute(fresh, f"import time; time.sleep(1); print('{tag}')"))

    t0 = time.monotonic()
    threads = [threading.Thread(target=run, args=(tag,)) for tag in ("first", "second")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert time.monotonic() - t0 >= 2.0
    assert sorted(r.stdout for r in results) == ["first\n", "second\n"]


def test_quota_per_user():
    svc = NotebookService(max_kernels_per_user=1)
    try:
        svc.start("alice")
        with pytest.raises(QuotaExceeded):
            svc.start("alice")
        bob = svc.start("bob")  # 别人的配额不受影响
        svc.shutdown(bob)
        svc.shutdown(svc.sessions("alice")[0])
        svc.shutdown(svc.start("alice"))  # 关掉以后可以重新申请
    finally:
        svc.shutdown_all()


def test_idle_kernels_are_culled_busy_ones_kept():
    svc = NotebookService(idle_timeout=60)
    try:
        idle, busy = svc.start("idle"), svc.start("busy")
        worker = threading.Thread(target=svc.execute, args=(busy, "import time; time.sleep(2)"))
        worker.start()
        time.sleep(0.5)
        culled = svc.cull_idle(now=time.monotonic() + 120)  # 模拟两分钟后
        worker.join()
        assert culled == [idle]
        assert svc.sessions() == [busy]
        assert svc.cull_idle() == []  # 刚执行完，不空闲
    finally:
        svc.shutdown_all()
