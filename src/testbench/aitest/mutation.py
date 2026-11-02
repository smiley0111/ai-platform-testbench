"""变异测试：往网关源码里植入真实可能出现的 bug，看用例能不能发现。

每个变异体只改一处代码。用例在原版网关上通过、在变异体上失败，就算“杀死”了这个变异体。
变异得分 = 被杀死的变异体 / 变异体总数，比覆盖率更能说明用例有没有用：
覆盖率只说明代码被执行过，变异得分说明出了 bug 用例真的会报。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from testbench.aitest.cases import Case, save_cases
from testbench.aitest.runner import Result

SRC = Path(__file__).resolve().parents[1]  # src/testbench


@dataclass
class Mutant:
    id: str
    text: str  # 模拟的 bug
    file: str  # 相对 src/testbench
    old: str
    new: str


MUTANTS = [
    Mutant("M01", "temperature 上界写成 3", "gateway/schemas.py", "ge=0, le=2)", "ge=0, le=3)"),
    Mutant("M02", "top_p 上界写成 1.5", "gateway/schemas.py", "top_p: float | None = Field(default=None, ge=0, le=1)",
           "top_p: float | None = Field(default=None, ge=0, le=1.5)"),
    Mutant("M03", "n 允许大于 1", "gateway/schemas.py", "Field(default=1, ge=1, le=1)", "Field(default=1, ge=1, le=4)"),
    Mutant("M04", "messages 允许为空", "gateway/schemas.py", "Field(min_length=1)", "Field(min_length=0)"),
    Mutant("M05", "max_tokens 上限差一（4096 被拒绝）", "gateway/app.py",
           "req.max_tokens > config.max_tokens_limit", "req.max_tokens >= config.max_tokens_limit"),
    Mutant("M06", "RBAC 失效：受限 Key 能访问所有模型", "gateway/app.py",
           "if policy.models is not None and name not in policy.models:", "if False:"),
    Mutant("M07", "模型类型不检查", "gateway/app.py", "if spec.kind != kind:", "if False:"),
    Mutant("M08", "内容过滤失效", "gateway/app.py", "return any(w in text for w in config.blocked_words)", "return False"),
    Mutant("M09", "图片输入不检查", "gateway/app.py", "if any(m.has_image() for m in req.messages) and not spec.vision:",
           "if False:"),
    Mutant("M10", "Key 无效时返回 403 而不是 401", "gateway/app.py",
           'raise APIError(401, "Invalid API key."', 'raise APIError(403, "Invalid API key."'),
    Mutant("M11", "非法 JSON 未处理（变成 500）", "gateway/app.py",
           "except (json.JSONDecodeError, UnicodeDecodeError) as e:", "except UnicodeDecodeError as e:"),
    Mutant("M12", "Embeddings 不拒绝空输入", "gateway/schemas.py", "if not items or any(not s for s in items):", "if False:"),
    Mutant("M13", "模型列表不鉴权", "gateway/app.py",
           "    async def list_models(request: Request) -> dict:\n        _, policy = authenticate(request)",
           "    async def list_models(request: Request) -> dict:\n        policy = KeyPolicy()"),
    Mutant("M14", "流式请求按非流式返回", "gateway/app.py", "        if not req.stream:\n", "        if True:\n"),
]


def _run_on(src_root: Path, cases_file: Path, out: Path) -> list[dict]:
    env = {**os.environ, "PYTHONPATH": str(src_root), "PYTHONUTF8": "1"}
    subprocess.run([sys.executable, "-m", "testbench.aitest.runner", str(cases_file), str(out)],
                   env=env, check=True, capture_output=True, text=True)
    return json.loads(out.read_text(encoding="utf-8"))


def mutation_score(cases: list[Case], baseline: list[Result]) -> dict[str, bool]:
    """返回 {变异体 id: 是否被杀死}。只用在原版上通过的用例，期望写错的用例不参与。"""
    valid = [c for c, r in zip(cases, baseline, strict=True) if r.passed]
    killed = {m.id: False for m in MUTANTS}
    if not valid:
        return killed
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cases_file = tmp / "cases.json"
        save_cases(valid, cases_file)
        for m in MUTANTS:
            root = tmp / m.id
            shutil.copytree(SRC, root / "testbench", ignore=shutil.ignore_patterns("__pycache__"))
            target = root / "testbench" / m.file
            code = target.read_text(encoding="utf-8")
            if code.count(m.old) != 1:
                raise RuntimeError(f"{m.id}: pattern found {code.count(m.old)} times in {m.file}")
            target.write_text(code.replace(m.old, m.new), encoding="utf-8")
            results = _run_on(root, cases_file, tmp / f"{m.id}.json")
            killed[m.id] = not all(r["passed"] for r in results)
    return killed
