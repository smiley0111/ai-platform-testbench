"""训练测试的公共工具。

每次训练都在子进程里跑：故障注入用 os._exit 模拟进程被杀，不能杀掉 pytest 自己；
多进程（spawn）在子进程里也更接近真实的训练作业。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CRASH_EXIT_CODE = 17
JOB_TIMEOUT = 180


@dataclass
class Run:
    out: Path
    returncode: int
    seconds: float
    stderr: str

    def losses(self) -> list[tuple[int, float]]:
        lines = (self.out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        return [(r["step"], r["loss"]) for r in map(json.loads, filter(None, lines))]

    def final(self, rank: int | None = None) -> dict:
        import torch

        name = "final.pt" if rank is None else f"final-rank{rank}.pt"
        return torch.load(self.out / name, weights_only=True)

    def summary(self) -> dict:
        return json.loads((self.out / "summary.json").read_text())

    def checkpoints(self) -> list[str]:
        root = self.out / "checkpoints"
        return sorted(p.name for p in root.iterdir()) if root.exists() else []


def train(out: Path, **opts) -> Run:
    """opts 对应 TrainConfig 字段，例如 world_size=2、crash_at_step=25、resume=False。"""
    args = [sys.executable, "-m", "testbench.training.trainer", "--out", str(out)]
    for k, v in opts.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            args.append(flag if v else "--no-" + k.replace("_", "-"))
        else:
            args += [flag, str(v)]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUTF8": "1"}
    t0 = time.perf_counter()
    p = subprocess.run(args, env=env, cwd=ROOT, capture_output=True, text=True, timeout=JOB_TIMEOUT)
    return Run(out, p.returncode, time.perf_counter() - t0, p.stderr)


def ok(run: Run) -> Run:
    assert run.returncode == 0, f"training failed ({run.returncode}):\n{run.stderr[-3000:]}"
    return run


def assert_same_params(a: dict, b: dict, exact: bool = True, atol: float = 1e-6) -> None:
    import torch

    assert a.keys() == b.keys()
    for k in a:
        if exact:
            assert torch.equal(a[k], b[k]), f"{k} differs, max diff {(a[k] - b[k]).abs().max():.3e}"
        else:
            assert torch.allclose(a[k], b[k], atol=atol), f"{k} max diff {(a[k] - b[k]).abs().max():.3e}"


def max_param_diff(a: dict, b: dict) -> float:
    return max((a[k] - b[k]).abs().max().item() for k in a)
