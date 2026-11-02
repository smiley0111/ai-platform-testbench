"""代码执行与输出：结果、状态保持、各类输出的捕获、错误处理、输出截断。"""

from __future__ import annotations

import pytest
from nbkit import OUTPUT_LIMIT

pytestmark = pytest.mark.notebook


def test_expression_result_and_execution_count(svc, shared):
    a = svc.execute(shared, "1 + 1")
    b = svc.execute(shared, "2 * 21")
    assert (a.status, a.result) == ("ok", "2")
    assert b.result == "42"
    assert b.execution_count == a.execution_count + 1


def test_state_persists_between_cells(svc, shared):
    svc.execute(shared, "counter = 0")
    for _ in range(3):
        svc.execute(shared, "counter += 1")
    assert svc.execute(shared, "counter").result == "3"


def test_stdout_and_stderr_are_separated(svc, shared):
    """ipykernel 给两个流各自缓冲、各自刷新：流内顺序有保证，两个流之间的先后没有保证，
    所以只断言各自的内容，不断言交错顺序。"""
    r = svc.execute(shared, "import sys\nprint('a')\nprint('b', file=sys.stderr)\nprint('c')")
    assert r.stdout == "a\nc\n"
    assert r.stderr == "b\n"


def test_statement_has_no_result(svc, shared):
    r = svc.execute(shared, "y = 5")
    assert r.status == "ok" and r.result is None and r.outputs == []


def test_rich_display_output(svc, shared):
    r = svc.execute(shared, "from IPython.display import HTML, display\ndisplay(HTML('<b>bold</b>'))")
    [out] = r.outputs
    assert out["output_type"] == "display_data"
    assert out["data"]["text/html"] == "<b>bold</b>"
    assert "text/plain" in out["data"]  # 前端不支持 HTML 时的降级显示


def test_unicode_output_roundtrip(svc, shared):
    r = svc.execute(shared, "print('中文输出 ✓')\n'返回值'")
    assert r.stdout == "中文输出 ✓\n"
    assert r.result == "'返回值'"


def test_runtime_error_is_reported_and_kernel_survives(svc, shared):
    svc.execute(shared, "keep = 'still here'")
    r = svc.execute(shared, "1 / 0")
    assert r.status == "error"
    assert r.error["ename"] == "ZeroDivisionError"
    assert r.error["traceback"], "traceback should not be empty"
    assert r.outputs[-1]["output_type"] == "error"
    assert not r.restarted
    assert svc.execute(shared, "keep").result == "'still here'"


def test_syntax_error(svc, shared):
    r = svc.execute(shared, "def broken(:")
    assert r.status == "error" and r.error["ename"] == "SyntaxError"


def test_output_before_error_is_kept(svc, shared):
    r = svc.execute(shared, "print('step 1')\nraise ValueError('boom')")
    assert r.stdout == "step 1\n"
    assert (r.error["ename"], r.error["evalue"]) == ("ValueError", "boom")


def test_output_flood_is_truncated(svc, shared):
    """一个 print 循环就能产生几十 MB 输出，不截断会撑爆服务端内存和浏览器。"""
    r = svc.execute(shared, "for i in range(20000):\n    print('line', i)")
    assert r.status == "ok"
    assert r.truncated
    assert len(r.stdout) == OUTPUT_LIMIT
    assert r.stdout.startswith("line 0\n")
    assert svc.execute(shared, "'responsive'").result == "'responsive'"


def test_small_output_is_not_truncated(svc, shared):
    r = svc.execute(shared, "print('x' * 100)")
    assert not r.truncated and len(r.stdout) == 101
