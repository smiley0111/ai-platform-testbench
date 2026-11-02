"""Notebook 测试的公共夹具。

内核启动一次要 3~6 秒：只读的用例共用一个内核（shared），会破坏内核状态的用例各自开新内核（fresh）。
整轮测试结束时检查本进程下没有残留的 ipykernel 子进程——孤儿内核是 Notebook 平台最常见的资源泄漏。
"""

from __future__ import annotations

import time
import uuid

import pytest
from nbkit import EXEC_TIMEOUT, INTERRUPT_GRACE, MEMORY_LIMIT_MB, OUTPUT_LIMIT, kernel_children

from testbench.notebook import NotebookService


@pytest.fixture(scope="session")
def svc():
    service = NotebookService(max_kernels_per_user=50, exec_timeout=EXEC_TIMEOUT,
                              interrupt_grace=INTERRUPT_GRACE, output_limit=OUTPUT_LIMIT,
                              memory_limit_mb=MEMORY_LIMIT_MB)
    yield service
    service.shutdown_all()
    deadline = time.monotonic() + 10
    while kernel_children() and time.monotonic() < deadline:
        time.sleep(0.2)
    leaked = kernel_children()
    assert not leaked, f"orphan kernels: {[p.pid for p in leaked]}"


@pytest.fixture(scope="module")
def shared(svc):
    sid = svc.start("shared")
    yield sid
    svc.shutdown(sid)


@pytest.fixture
def fresh(svc):
    sid = svc.start(f"user-{uuid.uuid4().hex[:6]}")
    yield sid
    if sid in svc.sessions():
        svc.shutdown(sid)
