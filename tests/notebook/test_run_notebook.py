"""整本执行（定时任务、批量跑 Notebook 的场景）：输出写回 .ipynb、遇错即停、允许出错继续、单元超时。"""

from __future__ import annotations

import nbformat
import pytest
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

pytestmark = pytest.mark.notebook


def make_nb(path, *cells):
    nb = new_notebook(cells=[new_markdown_cell("# 标题")] + [new_code_cell(c) for c in cells])
    nbformat.write(nb, path)
    return path


def code_cells(path):
    return [c for c in nbformat.read(path, as_version=4).cells if c.cell_type == "code"]


def test_run_writes_outputs_back(svc, fresh, tmp_path):
    src = make_nb(tmp_path / "in.ipynb", "x = 20", "print(x * 2)", "x + 1")
    dst = tmp_path / "out.ipynb"
    report = svc.run_notebook(fresh, src, dst)
    assert report.ok and report.cells_run == 3
    nb = nbformat.read(dst, as_version=4)
    nbformat.validate(nb)  # 写回的文件必须是合法的 Notebook，Jupyter 才能打开
    cells = code_cells(dst)
    assert [c.execution_count for c in cells] == [1, 2, 3]
    assert cells[1].outputs[0]["text"] == "40\n"
    assert cells[2].outputs[0]["data"]["text/plain"] == "21"
    assert nb.cells[0].source == "# 标题"  # Markdown 单元原样保留


def test_run_stops_at_first_error(svc, fresh, tmp_path):
    src = make_nb(tmp_path / "in.ipynb", "a = 1", "1 / 0", "print('should not run')")
    dst = tmp_path / "out.ipynb"
    report = svc.run_notebook(fresh, src, dst)
    assert not report.ok
    assert (report.failed_cell, report.failure, report.cells_run) == (2, "error", 2)
    cells = code_cells(dst)
    assert cells[1].outputs[0]["ename"] == "ZeroDivisionError"
    assert cells[2].outputs == [] and cells[2].execution_count is None


def test_allow_errors_continues(svc, fresh, tmp_path):
    src = make_nb(tmp_path / "in.ipynb", "1 / 0", "print('ran anyway')")
    dst = tmp_path / "out.ipynb"
    report = svc.run_notebook(fresh, src, dst, allow_errors=True)
    assert not report.ok and report.failed_cell == 1 and report.cells_run == 2
    assert code_cells(dst)[1].outputs[0]["text"] == "ran anyway\n"


def test_cell_timeout_is_reported(svc, fresh, tmp_path):
    src = make_nb(tmp_path / "in.ipynb", "import time", "while True:\n    time.sleep(0.05)", "print('after')")
    dst = tmp_path / "out.ipynb"
    report = svc.run_notebook(fresh, src, dst, allow_errors=True, cell_timeout=1)
    assert (report.failed_cell, report.failure) == (2, "timeout")
    assert report.cells_run == 2  # 超时不属于“允许的错误”，后面的单元不再执行
