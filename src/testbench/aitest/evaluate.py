"""评测 AI 生成用例：格式合法率、有效率、需求覆盖率、变异得分，并与人工用例对比，输出报告。

    python -m testbench.aitest.evaluate                                  # 默认 5 组
    python -m testbench.aitest.evaluate --runs qwen2.5:3b/schema,qwen2.5:3b/free

生成结果缓存在 reports/aitest/raw/，重跑报告不会重新调用模型；加 --force 重新生成。
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from testbench.aitest.cases import Case, load_cases, parse_output, save_cases
from testbench.aitest.generate import CATEGORIES, generate
from testbench.aitest.mutation import MUTANTS, mutation_score
from testbench.aitest.requirements import REQUIREMENTS, coverage
from testbench.aitest.runner import Result, run_cases

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports" / "aitest"
GOLDEN = Path(__file__).parent / "golden_cases.json"
DEFAULT_RUNS = ("qwen2.5:0.5b/schema,qwen2.5:0.5b/free,qwen2.5:1.5b/schema,qwen2.5:1.5b/free,"
                "qwen2.5:3b/schema")


@dataclass
class Evaluation:
    label: str
    seconds: float = 0.0
    output_tokens: int = 0
    generated: int = 0
    malformed: list[str] = field(default_factory=list)
    duplicates: int = 0
    cases: list[Case] = field(default_factory=list)
    results: list[Result] = field(default_factory=list)
    covered: dict[str, bool] = field(default_factory=dict)
    killed: dict[str, bool] = field(default_factory=dict)

    @property
    def valid(self) -> int:
        return sum(r.passed for r in self.results)

    @property
    def valid_rate(self) -> float:
        return self.valid / len(self.cases) if self.cases else 0.0


def dedup(cases: list[Case]) -> tuple[list[Case], int]:
    seen, out = set(), []
    for c in cases:
        if c.key() not in seen:
            seen.add(c.key())
            out.append(c)
    return out, len(cases) - len(out)


def score(ev: Evaluation) -> Evaluation:
    ev.results = run_cases(ev.cases)
    ev.covered = coverage(ev.cases, ev.results)
    ev.killed = mutation_score(ev.cases, ev.results)
    return ev


def evaluate_model(model: str, schema: bool, force: bool) -> Evaluation:
    ev = Evaluation(f"{model} {'约束解码' if schema else '无约束'}")
    raw_dir = OUT / "raw"
    all_cases: list[Case] = []
    for cat in CATEGORIES:
        if force:
            (raw_dir / f"{model.replace(':', '_')}-{'schema' if schema else 'free'}-{cat}.json").unlink(missing_ok=True)
        g = generate(model, cat, schema, cache_dir=raw_dir)
        print(f"[gen] {model} schema={schema} {cat}: {g.seconds:.0f}s, {g.output_tokens} tokens", flush=True)
        ev.seconds += g.seconds
        ev.output_tokens += g.output_tokens
        cases, errors = parse_output(g.content, source=cat)
        ev.generated += len(cases) + len(errors)
        ev.malformed += errors
        all_cases += cases
    ev.cases, ev.duplicates = dedup(all_cases)
    save_cases(ev.cases, OUT / f"cases-{model.replace(':', '_')}-{'schema' if schema else 'free'}.json")
    return score(ev)


def golden() -> Evaluation:
    cases = load_cases(GOLDEN)
    ev = Evaluation("人工编写（基准）", generated=len(cases))
    ev.cases = cases
    return score(ev)


# ---------- 报告 ----------

def pct(a: int, b: int) -> str:
    return f"{a / b:.0%}" if b else "-"


def mismatch_patterns(ev: Evaluation) -> Counter:
    """期望写错的用例按“期望 → 实际”归类，看模型常犯哪类错。"""
    return Counter(f"期望 {r.expect_status} {r.expect_code or ''} → 实际 {r.status} {r.code or ''}".replace("  ", " ")
                   for r in ev.results if not r.passed)


def write_report(evs: list[Evaluation]) -> Path:
    n_req, n_mut = len(REQUIREMENTS), len(MUTANTS)
    L = ["# AI 辅助测试评测：大模型根据接口文档生成用例", "",
         "被测对象是阶段 1 的推理网关（Mock 后端）。模型只拿到 `src/testbench/aitest/api_spec.md` 这份接口文档，"
         "按 5 个测试类别分别生成用例（每类要求 8 条），输出为 JSON 数据，由执行器统一发请求和比对。", "",
         "## 指标说明", "",
         "- **格式不合法**：缺字段、路径不在文档里、POST 没有请求体等，执行前就被丢弃",
         "- **有效**：在正确实现的网关上执行通过，即期望结果写对了。期望写错的用例在回归里只会制造误报",
         f"- **需求覆盖**：接口文档拆成 {n_req} 条需求点，被至少一条有效用例命中才算覆盖（按实际行为判断，不看用例名字）",
         f"- **变异得分**：往网关里植入 {n_mut} 个真实可能出现的 bug（每次只改一处），有效用例能发现几个", "",
         "## 总览", "",
         "| 来源 | 生成耗时 | 输出 token | 生成条数 | 格式不合法 | 重复 | 执行条数 | 有效 | 有效率 | 需求覆盖 | 变异得分 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ev in evs:
        L.append(f"| {ev.label} | {ev.seconds:.0f}s | {ev.output_tokens} | {ev.generated} | {len(ev.malformed)} | "
                 f"{ev.duplicates} | {len(ev.cases)} | {ev.valid} | {ev.valid_rate:.0%} | "
                 f"{sum(ev.covered.values())}/{n_req}（{pct(sum(ev.covered.values()), n_req)}） | "
                 f"{sum(ev.killed.values())}/{n_mut}（{pct(sum(ev.killed.values()), n_mut)}） |")

    L += ["", "## 需求覆盖明细", "", "| 需求 | " + " | ".join(ev.label for ev in evs) + " |",
          "|---|" + "---|" * len(evs)]
    for req in REQUIREMENTS:
        L.append(f"| {req.id} {req.text} | " + " | ".join("✓" if ev.covered.get(req.id) else "" for ev in evs) + " |")

    L += ["", "## 变异体明细（✓ = 被发现）", "", "| 植入的 bug | " + " | ".join(ev.label for ev in evs) + " |",
          "|---|" + "---|" * len(evs)]
    for m in MUTANTS:
        L.append(f"| {m.id} {m.text} | " + " | ".join("✓" if ev.killed.get(m.id) else "" for ev in evs) + " |")

    L += ["", "## 期望写错的用例：常见模式", ""]
    for ev in evs:
        pats = mismatch_patterns(ev)
        if not pats:
            continue
        L += [f"**{ev.label}**（共 {sum(pats.values())} 条）", ""]
        L += [f"- {p}：{n} 条" for p, n in pats.most_common(6)]
        L.append("")
    L += ["## 格式不合法的原因", ""]
    for ev in evs:
        if ev.malformed:
            L += [f"**{ev.label}**：" + "；".join(f"{k}（{v}）" for k, v in Counter(ev.malformed).most_common(5)), ""]
    L += ["## 原始数据", "", "`reports/aitest/raw/` 是模型的原始输出，`reports/aitest/cases-*.json` 是解析去重后的用例。", ""]
    path = ROOT / "reports" / "aitest-report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps([{
        "label": ev.label, "generated": ev.generated, "malformed": len(ev.malformed), "executed": len(ev.cases),
        "valid": ev.valid, "covered": sum(ev.covered.values()), "killed": sum(ev.killed.values()),
        "seconds": round(ev.seconds, 1)} for ev in evs], ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=DEFAULT_RUNS, help="模型/模式，逗号分隔；模式为 schema（约束解码）或 free")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t0 = time.monotonic()
    evs = [golden()]
    for run in args.runs.split(","):
        model, mode = run.rsplit("/", 1)
        evs.append(evaluate_model(model, mode == "schema", args.force))
    print(f"report: {write_report(evs)} ({time.monotonic() - t0:.0f}s)")


if __name__ == "__main__":
    main()
