"""异常与资源限制：超时中断、中断无效时升级重启、用户手动停止、内核崩溃、内存超限。

平台对用户的承诺：任何代码都不能让用户无限等待；内核出事要能自动恢复；
状态丢没丢要明确告诉用户（restarted 字段），不能悄悄换一个空内核。
"""

from __future__ import annotations

import threading
import time

import pytest
from nbkit import MEMORY_LIMIT_MB, WORST_CASE_EXTRA

pytestmark = pytest.mark.notebook

TIMEOUT = 1.0


def test_timeout_interrupts_and_keeps_state(svc, fresh):
    svc.execute(fresh, "data = [1, 2, 3]")
    r = svc.execute(fresh, "import time\nwhile True:\n    time.sleep(0.05)", timeout=TIMEOUT)
    assert r.status == "timeout"
    assert r.error["ename"] == "KeyboardInterrupt"
    assert not r.restarted
    assert r.duration < TIMEOUT + 1.5
    assert svc.execute(fresh, "data").result == "[1, 2, 3]"


def test_swallowed_interrupt_escalates_to_restart(svc, fresh):
    """用户代码捕获了 KeyboardInterrupt，中断无效，宽限期过后必须强制重启，并告知状态已丢失。"""
    svc.execute(fresh, "data = 1")
    code = "import time\nwhile True:\n    try:\n        time.sleep(0.05)\n    except KeyboardInterrupt:\n        pass"
    r = svc.execute(fresh, code, timeout=TIMEOUT)
    assert r.status == "timeout"
    assert r.restarted
    assert r.duration < TIMEOUT + WORST_CASE_EXTRA
    assert svc.execute(fresh, "data").error["ename"] == "NameError"


def test_blocking_native_call_is_bounded(svc, fresh):
    """长时间阻塞在 C 层（如一次 sleep 60 秒）时，Windows 上的中断要等它返回才生效，
    Linux 上会立即打断。不管走哪条路，用户的等待时间都必须有上限，之后内核可用。"""
    r = svc.execute(fresh, "import time; time.sleep(60)", timeout=TIMEOUT)
    assert r.status == "timeout"
    assert r.duration < TIMEOUT + WORST_CASE_EXTRA
    assert r.restarted or r.error["ename"] == "KeyboardInterrupt"
    assert svc.execute(fresh, "'usable'").result == "'usable'"


def test_user_stop_button(svc, fresh):
    """用户在执行过程中点“停止”：返回 KeyboardInterrupt 错误，内核保留。"""
    svc.execute(fresh, "kept = True")
    threading.Timer(0.5, svc.interrupt, args=(fresh,)).start()
    r = svc.execute(fresh, "import time\nfor _ in range(200):\n    time.sleep(0.05)")
    assert r.status == "error" and r.error["ename"] == "KeyboardInterrupt"
    assert r.duration < 3
    assert svc.execute(fresh, "kept").result == "True"


def test_kernel_crash_is_detected_and_recovered(svc, fresh):
    svc.execute(fresh, "data = 1")
    old_pid = svc.pid(fresh)
    r = svc.execute(fresh, "import os; os._exit(1)")
    assert r.status == "dead"
    assert r.restarted
    assert r.duration < 10
    assert svc.pid(fresh) != old_pid
    assert svc.execute(fresh, "data").error["ename"] == "NameError"
    assert svc.execute(fresh, "1 + 1").result == "2"


def test_crash_after_partial_output_keeps_that_output(svc, fresh):
    r = svc.execute(fresh, "import os, sys, time\nprint('before crash'); sys.stdout.flush()\ntime.sleep(0.5)\nos._exit(3)")
    assert r.status == "dead"
    assert r.stdout == "before crash\n"


def test_memory_limit_restarts_kernel(svc, fresh):
    alloc_mb = MEMORY_LIMIT_MB + 300
    code = f"buf = bytearray({alloc_mb} * 1024 * 1024)\nimport time; time.sleep(5)"
    r = svc.execute(fresh, code)
    assert r.status == "oom"
    assert r.restarted
    assert r.duration < 10
    assert svc.execute(fresh, "'recovered'").result == "'recovered'"


def test_memory_below_limit_is_fine(svc, fresh):
    r = svc.execute(fresh, f"buf = bytearray({MEMORY_LIMIT_MB // 4} * 1024 * 1024)\nlen(buf) > 0")
    assert (r.status, r.result) == ("ok", "True")


def test_service_keeps_working_after_many_faults(svc, fresh):
    """连续出事后服务和内核仍正常，没有句柄或线程泄漏导致的卡死。"""
    for code in ["1/0", "import os; os._exit(1)", "1/0"]:
        svc.execute(fresh, code, timeout=TIMEOUT)
    t0 = time.monotonic()
    assert svc.execute(fresh, "40 + 2").result == "42"
    assert time.monotonic() - t0 < 1
