"""Notebook 测试的常量和工具函数（单独成模块，避免和其他目录的 conftest 重名）。"""

from __future__ import annotations

import psutil

EXEC_TIMEOUT = 10.0
INTERRUPT_GRACE = 2.0
OUTPUT_LIMIT = 10_000
MEMORY_LIMIT_MB = 500
# 超时后用户最多等多久：超时 + 中断宽限 + 重启内核（Windows 上约 3~4 秒）
WORST_CASE_EXTRA = INTERRUPT_GRACE + 8.0


def kernel_children() -> list[psutil.Process]:
    out = []
    for p in psutil.Process().children(recursive=True):
        try:
            if "ipykernel" in " ".join(p.cmdline()):
                out.append(p)
        except psutil.Error:
            pass
    return out
