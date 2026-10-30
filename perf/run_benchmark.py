"""一键跑完性能基准并生成 reports/perf-report.md。

场景：
  1. 冷启动 vs 热启动：卸载模型后测首 token 时延，再连续测热请求
  2. 阶梯并发：1/2/4/8/16 并发闭环压测，找吞吐拐点
  3. 负载下限流：小配额 key 压测，实测放行数对比令牌桶理论值
  4. 负载下超时：网关超时调小后高并发，观察 504、熔断与压后恢复

网关和 Locust 都以子进程启动：Locust 依赖 gevent 打补丁，不能和 uvicorn 放在同一进程。

    python perf/run_benchmark.py            # 完整跑，约 10 分钟
    python perf/run_benchmark.py --quick    # 每档 15 秒，用来验证脚本
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
RAW = REPORTS / "raw"
OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
BACKEND_MODEL = os.getenv("TESTBENCH_CHAT_MODEL", "qwen2.5:0.5b")
API_KEY = "sk-perf"
MAX_TOKENS = 64


def http() -> httpx.Client:
    return httpx.Client(trust_env=False, timeout=180)


# ---------- 统计 ----------

def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(p / 100 * len(s) + 0.5) - 1))
    return s[k]


def fmt(v: float | None, unit: str = "s", nd: int = 2) -> str:
    return "-" if v is None else f"{v:.{nd}f}{unit}"


@dataclass
class Summary:
    label: str
    users: int
    duration: float
    rows: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> list[dict]:
        return [r for r in self.rows if r["status"] == 200 and r["code"] is None]

    def latency(self, key: str, p: float, rows: list[dict] | None = None) -> float | None:
        return pct([r[key] for r in (rows if rows is not None else self.ok) if r[key] is not None], p)

    @property
    def rps(self) -> float:
        return len(self.ok) / self.duration

    @property
    def tokens_per_s(self) -> float:
        return sum(r["completion_tokens"] for r in self.ok) / self.duration

    @property
    def error_rate(self) -> float:
        return 0.0 if not self.rows else 1 - len(self.ok) / len(self.rows)

    @property
    def codes(self) -> Counter:
        return Counter(r["code"] or str(r["status"]) for r in self.rows)


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------- 进程管理 ----------

class Gateway:
    def __init__(self, port: int, rpm: int = 1_000_000, timeout: float = 120):
        self.port, self.url = port, f"http://127.0.0.1:{port}"
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "TESTBENCH_BACKEND": "ollama",
               "TESTBENCH_API_KEY": API_KEY, "TESTBENCH_RPM": str(rpm), "TESTBENCH_TIMEOUT": str(timeout),
               "TESTBENCH_PORT": str(port), "TESTBENCH_LOG_LEVEL": "warning", "OLLAMA_URL": OLLAMA}
        self.proc = subprocess.Popen(
            [sys.executable, "-c", "from testbench.gateway.app import main; main()"], env=env, cwd=ROOT)
        with http() as c:
            for _ in range(100):
                try:
                    if c.get(f"{self.url}/healthz").status_code == 200:
                        return
                except httpx.TransportError:
                    pass
                time.sleep(0.1)
        self.stop()
        raise RuntimeError("gateway did not start")

    def stop(self) -> None:
        self.proc.terminate()
        self.proc.wait(10)

    def __enter__(self) -> Gateway:
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def run_locust(host: str, users: int, seconds: int, label: str, max_tokens: int = MAX_TOKENS) -> Summary:
    records = RAW / f"{label}.jsonl"
    records.unlink(missing_ok=True)
    # Locust 会读 pyproject.toml，Windows 默认编码解不了中文，强制 UTF-8
    env = {**os.environ, "PYTHONUTF8": "1", "PERF_RECORDS": str(records), "PERF_LABEL": label,
           "TESTBENCH_API_KEY": API_KEY, "PERF_MAX_TOKENS": str(max_tokens)}
    cmd = [sys.executable, "-m", "locust", "-f", str(ROOT / "perf" / "locustfile.py"), "--headless",
           "--host", host, "-u", str(users), "-r", str(users), "-t", f"{seconds}s",
           "--csv", str(RAW / label), "--only-summary", "--loglevel", "WARNING"]
    print(f"[locust] {label}: {users} users x {seconds}s", flush=True)
    # 有失败请求时 Locust 退出码为 1，属于正常情况，所以用"有没有明细"判断是否真的跑起来了
    proc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                          text=True, encoding="utf-8", errors="replace", check=False)
    rows = load_rows(records)
    if not rows:
        raise RuntimeError(f"locust produced no records for {label}:\n{proc.stderr[-2000:]}")
    return Summary(label, users, seconds, rows)


def single_stream(c: httpx.Client, base: str) -> dict:
    body = {"model": "qwen-mini", "messages": [{"role": "user", "content": "用三百字介绍一下 Kubernetes。"}],
            "stream": True, "max_tokens": MAX_TOKENS, "temperature": 0,
            "stream_options": {"include_usage": True}}
    t0, ttft, tokens = time.perf_counter(), None, 0
    with c.stream("POST", f"{base}/v1/chat/completions", json=body,
                  headers={"Authorization": f"Bearer {API_KEY}"}) as r:
        if r.status_code != 200:
            r.read()
            return {"status": r.status_code, "ttft": None, "e2e": time.perf_counter() - t0, "tokens": 0}
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                tokens = chunk["usage"]["completion_tokens"]
            if ttft is None and any(ch.get("delta", {}).get("content") for ch in chunk.get("choices", [])):
                ttft = time.perf_counter() - t0
    return {"status": 200, "ttft": ttft, "e2e": time.perf_counter() - t0, "tokens": tokens}


def unload_model(c: httpx.Client) -> None:
    c.post(f"{OLLAMA}/api/generate", json={"model": BACKEND_MODEL, "keep_alive": 0})
    for _ in range(50):
        if not c.get(f"{OLLAMA}/api/ps").json().get("models"):
            return
        time.sleep(0.2)


# ---------- 场景 ----------

def scenario_cold_warm(rounds: int) -> dict:
    cold, warm = [], []
    with Gateway(8101) as gw, http() as c:
        for i in range(rounds):
            unload_model(c)
            cold.append(single_stream(c, gw.url))
            print(f"[cold] round {i + 1}: ttft={cold[-1]['ttft']:.2f}s", flush=True)
        for _ in range(rounds * 2):
            warm.append(single_stream(c, gw.url))
    return {"cold": cold, "warm": warm}


def scenario_ladder(levels: list[int], seconds: int) -> list[Summary]:
    out = []
    with Gateway(8102) as gw, http() as c:
        single_stream(c, gw.url)  # 预热，排除冷启动对第一档的干扰
        for n in levels:
            out.append(run_locust(gw.url, n, seconds, f"ladder-u{n}"))
    return out


def scenario_rate_limit(rpm: int, users: int, seconds: int) -> tuple[Summary, dict]:
    # 短回答让模型不成为瓶颈，放行数才能和令牌桶理论值对上
    with Gateway(8103, rpm=rpm) as gw:
        s = run_locust(gw.url, users, seconds, "ratelimit", max_tokens=8)
    expected = rpm + rpm / 60 * seconds
    return s, {"rpm": rpm, "expected_max": expected}


def scenario_timeout(timeout: float, users: int, seconds: int) -> tuple[Summary, dict]:
    with Gateway(8104, timeout=timeout) as gw, http() as c:
        single_stream(c, gw.url)
        s = run_locust(gw.url, users, seconds, "timeout")
        time.sleep(11)  # 熔断默认 10 秒后半开
        after = single_stream(c, gw.url)  # 压测停止后单请求，验证后端和熔断都已恢复
    return s, {"timeout": timeout, "chain": 2, "after": after}  # qwen-mini → qwen-backup


# ---------- 报告 ----------

def env_info() -> list[str]:
    with http() as c:
        try:
            ver = c.get(f"{OLLAMA}/api/version").json().get("version")
        except httpx.HTTPError:
            ver = "unknown"
    return [
        f"- 机器：{platform.platform()}，{os.cpu_count()} 逻辑核，无 GPU（CPU 推理）",
        f"- 后端：Ollama {ver}，模型 `{BACKEND_MODEL}`，`OLLAMA_NUM_PARALLEL`="
        f"`{os.getenv('OLLAMA_NUM_PARALLEL') or '默认'}`",
        f"- 请求：流式 `/v1/chat/completions`，`max_tokens={MAX_TOKENS}`，`temperature=0`，提示词要求长回答",
        "- 压测机与模型在同一台机器上，Locust 自身也占 CPU，绝对数值只代表本机",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M')}",
    ]


def find_knee(ladder: list[Summary]) -> Summary | None:
    """吞吐增长不足 10% 的第一档的前一档，视为饱和点（再加并发只会拉长排队）。"""
    for prev, cur in zip(ladder, ladder[1:]):
        if cur.tokens_per_s < prev.tokens_per_s * 1.10:
            return prev
    return None


def mermaid_chart(title: str, xs: list, y_label: str, bars: list[float], line: list[float]) -> list[str]:
    return ["```mermaid", "xychart-beta", f'  title "{title}"', f"  x-axis [{', '.join(map(str, xs))}]",
            f'  y-axis "{y_label}"', f"  bar [{', '.join(f'{v:.1f}' for v in bars)}]",
            f"  line [{', '.join(f'{v:.1f}' for v in line)}]", "```"]


def write_report(cw: dict, ladder: list[Summary], rl: tuple[Summary, dict], to: tuple[Summary, dict]) -> Path:
    L = ["# 性能基准报告：OpenAI 兼容网关 + Ollama", "", "## 环境", "", *env_info(), ""]

    cold_ttft = [r["ttft"] for r in cw["cold"]]
    warm_ttft = [r["ttft"] for r in cw["warm"]]
    warm_tps = [r["tokens"] / (r["e2e"] - r["ttft"]) for r in cw["warm"] if r["e2e"] > r["ttft"]]
    L += ["## 1. 冷启动 vs 热启动（单请求）", "",
          "| | 样本数 | 首 token p50 | 首 token 最大 | 端到端 p50 |", "|---|---|---|---|---|",
          f"| 冷启动（先卸载模型） | {len(cold_ttft)} | {fmt(pct(cold_ttft, 50))} | {fmt(max(cold_ttft))} | "
          f"{fmt(pct([r['e2e'] for r in cw['cold']], 50))} |",
          f"| 热启动 | {len(warm_ttft)} | {fmt(pct(warm_ttft, 50))} | {fmt(max(warm_ttft))} | "
          f"{fmt(pct([r['e2e'] for r in cw['warm']], 50))} |", "",
          f"冷启动首 token 约是热启动的 **{pct(cold_ttft, 50) / pct(warm_ttft, 50):.0f} 倍**，"
          f"多出的时间是把模型权重加载进内存。热启动单路生成速度约 **{pct(warm_tps, 50):.0f} token/s**。", ""]

    L += ["## 2. 阶梯并发（闭环，每档固定时长）", "",
          "| 并发 | 成功请求 | RPS | 吞吐 token/s | TPM | 首 token p50 | 首 token p95 | 端到端 p95 | 错误率 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in ladder:
        L.append(f"| {s.users} | {len(s.ok)} | {s.rps:.2f} | {s.tokens_per_s:.1f} | {s.tokens_per_s * 60:.0f} | "
                 f"{fmt(s.latency('ttft', 50))} | {fmt(s.latency('ttft', 95))} | {fmt(s.latency('e2e', 95))} | "
                 f"{s.error_rate:.1%} |")
    L += ["", "用 Little 定律校验（并发数 = 吞吐 × 时延，所以时延 ≈ 并发数 / RPS）：", "",
          "| 并发 | 预测时延 N/RPS | 实测端到端 p50 |", "|---|---|---|",
          *[f"| {s.users} | {s.users / s.rps:.2f}s | {fmt(s.latency('e2e', 50))} |" for s in ladder if s.rps],
          "", "两者吻合说明请求都在系统里排队，没有被丢弃或卡死；吞吐已经到顶，时延只随并发线性增长。"]
    L += ["", *mermaid_chart("吞吐(柱, token/s) 与首 token p95(线, 0.1s)", [s.users for s in ladder],
                             "token/s | 0.1s", [s.tokens_per_s for s in ladder],
                             [(s.latency("ttft", 95) or 0) * 10 for s in ladder]), ""]
    knee = find_knee(ladder)
    if knee:
        nxt = ladder[ladder.index(knee) + 1]
        L += [f"**饱和点约在 {knee.users} 并发**：再加到 {nxt.users} 并发，吞吐只从 {knee.tokens_per_s:.1f} 变到 "
              f"{nxt.tokens_per_s:.1f} token/s，首 token p95 却从 {fmt(knee.latency('ttft', 95))} 涨到 "
              f"{fmt(nxt.latency('ttft', 95))}。超过这个点，多出来的请求只是在后端排队。", ""]
    else:
        L += ["测试范围内吞吐一直在增长，没有出现明显饱和点，可以继续加大并发。", ""]

    s, meta = rl
    admitted, limited = len(s.ok), s.codes.get("rate_limit_exceeded", 0)
    lim_rows = [r for r in s.rows if r["code"] == "rate_limit_exceeded"]
    L += ["## 3. 负载下限流", "",
          f"API Key 配额 `rpm={meta['rpm']}`（令牌桶容量 {meta['rpm']}，每秒回填 {meta['rpm'] / 60:.2f}），"
          f"{s.users} 并发压 {s.duration:.0f} 秒，`max_tokens=8`，客户端遵守 `Retry-After` 退避。", "",
          "| 指标 | 值 |", "|---|---|",
          f"| 总请求 | {len(s.rows)} |",
          f"| 放行（200） | {admitted} |",
          f"| 理论最多放行 | {meta['expected_max']:.0f}（容量 + 回填速度 × 时长） |",
          f"| 被限流（429） | {limited}（{limited / max(1, len(s.rows)):.1%}） |",
          f"| 429 响应 p95 | {fmt((s.latency('e2e', 95, lim_rows) or 0) * 1000, 'ms', 1)} |",
          f"| 其他错误 | {sum(v for k, v in s.codes.items() if k not in ('200', 'rate_limit_exceeded'))} |", "",
          f"放行数 {'不超过' if admitted <= meta['expected_max'] + 1 else '**超过了**'}理论上限，"
          "被拒请求在网关直接返回，不占用模型算力。", ""]

    s, meta = to
    codes = s.codes
    bound = meta["timeout"] * meta["chain"]
    timeout_rows = [r for r in s.rows if r["code"] == "upstream_timeout"]
    L += ["## 4. 负载下超时", "",
          f"网关超时调到 `{meta['timeout']}s`（只约束首个 chunk），{s.users} 并发压 {s.duration:.0f} 秒，"
          "客户端出错后退避 1 秒。", "",
          "| 结果 | 次数 |", "|---|---|", *[f"| {k} | {v} |" for k, v in codes.most_common()], "",
          f"504 请求的端到端 p50 {fmt(s.latency('e2e', 50, timeout_rows))}、最大 "
          f"{fmt(max((r['e2e'] for r in timeout_rows), default=None))}。`qwen-mini` 配了降级模型，主模型超时后"
          f"会再试降级模型，所以客户端最坏要等 **超时 × 降级链长度 = {meta['timeout']} × {meta['chain']} = "
          f"{bound:.1f}s**，而不是配置里的 {meta['timeout']}s。设置超时要按整条链算。", "",
          f"压测停止 11 秒后（熔断半开窗口已过）单请求：状态 {meta['after']['status']}，"
          f"首 token {fmt(meta['after']['ttft'])}，端到端 {fmt(meta['after']['e2e'])}"
          "（与热启动对比即可判断后端是否已恢复）。", ""]
    if codes.get("circuit_open"):
        L += ["出现了 `circuit_open`：连续超时触发了熔断，网关直接拒绝后续请求，等于自动卸掉了一部分负载。", ""]

    L += ["## 原始数据", "", "`reports/raw/` 下每个场景有 Locust 的 CSV 和逐请求 JSONL。", ""]
    path = REPORTS / "perf-report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="每档 15 秒，只验证脚本")
    ap.add_argument("--levels", default="1,2,4,8,16")
    ap.add_argument("--report-only", action="store_true", help="不压测，用 reports/raw 里上次的数据重新生成报告")
    args = ap.parse_args()
    meta_path = RAW / "meta.json"
    if args.report_only:
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        ladder = [Summary(f"ladder-u{n}", n, m["seconds"], load_rows(RAW / f"ladder-u{n}.jsonl"))
                  for n in m["levels"]]
        rl = (Summary("ratelimit", m["rl"]["users"], m["rl"]["seconds"], load_rows(RAW / "ratelimit.jsonl")),
              m["rl"]["meta"])
        to = (Summary("timeout", m["to"]["users"], m["to"]["seconds"], load_rows(RAW / "timeout.jsonl")),
              m["to"]["meta"])
        print(f"report: {write_report(m['cold_warm'], ladder, rl, to)}")
        return

    if not shutil.which("locust") and subprocess.run([sys.executable, "-m", "locust", "--version"],
                                                     capture_output=True).returncode:
        sys.exit("locust not installed: pip install locust")
    RAW.mkdir(parents=True, exist_ok=True)
    seconds = 15 if args.quick else 60
    levels = [int(x) for x in args.levels.split(",")]

    cw = scenario_cold_warm(2 if args.quick else 3)
    ladder = scenario_ladder(levels, seconds)
    rl = scenario_rate_limit(rpm=12, users=8, seconds=30 if not args.quick else 15)
    to = scenario_timeout(timeout=1.5, users=16, seconds=30 if not args.quick else 15)
    meta_path.write_text(json.dumps({
        "seconds": seconds, "levels": levels, "cold_warm": cw,
        "rl": {"users": rl[0].users, "seconds": rl[0].duration, "meta": rl[1]},
        "to": {"users": to[0].users, "seconds": to[0].duration, "meta": to[1]},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report: {write_report(cw, ladder, rl, to)}")


if __name__ == "__main__":
    main()
