"""流式对话压测用户。

每个请求额外上报两条 Locust 指标（TTFT、E2E），并把明细写到 PERF_RECORDS 指定的 JSONL，
供 run_benchmark.py 汇总 p50/p95、吞吐和错误码分布。

单独使用：
    locust -f perf/locustfile.py --host http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import os
import random
import time

from locust import HttpUser, constant, task

API_KEY = os.getenv("TESTBENCH_API_KEY", "sk-local-dev")
MODEL = os.getenv("PERF_MODEL", "qwen-mini")
MAX_TOKENS = int(os.getenv("PERF_MAX_TOKENS", "64"))
LABEL = os.getenv("PERF_LABEL", "")
ERROR_BACKOFF = float(os.getenv("PERF_ERROR_BACKOFF", "1"))

# 提示词都要求长回答，保证每次都能生成到 max_tokens，输出长度稳定，不同并发档之间才可比
PROMPTS = [
    "用三百字介绍一下 Kubernetes 的调度过程。",
    "用三百字解释什么是大模型推理里的 KV Cache。",
    "用三百字说明 HTTP 长连接和 SSE 的区别。",
    "用三百字介绍 PyTorch 分布式数据并行的原理。",
]

_records = open(os.environ["PERF_RECORDS"], "a", encoding="utf-8") if os.getenv("PERF_RECORDS") else None


def _record(row: dict) -> None:
    if _records:
        _records.write(json.dumps(row, ensure_ascii=False) + "\n")
        _records.flush()


class ChatStreamUser(HttpUser):
    wait_time = constant(0)  # 闭环压测：上一个请求结束立刻发下一个，并发数 = 用户数

    def _fire(self, name: str, seconds: float, length: int = 0) -> None:
        self.environment.events.request.fire(
            request_type="STREAM", name=name, response_time=seconds * 1000,
            response_length=length, exception=None, context={},
        )

    @task
    def chat_stream(self) -> None:
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": random.choice(PROMPTS)}],
            "stream": True,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "stream_options": {"include_usage": True},
        }
        row = {"label": LABEL, "users": self.environment.runner.user_count, "start": time.time(),
               "status": None, "code": None, "ttft": None, "e2e": None, "completion_tokens": 0}
        t0 = time.perf_counter()
        with self.client.post("/v1/chat/completions", json=body, stream=True, catch_response=True,
                              headers={"Authorization": f"Bearer {API_KEY}"},
                              name="/v1/chat/completions", timeout=180) as resp:
            row["status"] = resp.status_code
            if resp.status_code != 200:
                try:
                    row["code"] = resp.json()["error"]["code"]
                except Exception:
                    row["code"] = "unparseable"
                row["e2e"] = time.perf_counter() - t0
                resp.failure(f"{resp.status_code} {row['code']}")
                _record(row)
                # 像 SDK 一样退避：有 Retry-After 就照做，否则等 1 秒，避免错误请求空转刷屏
                time.sleep(float(resp.headers.get("retry-after", ERROR_BACKOFF)))
                return

            done = False
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                data = raw[len("data: "):]
                if data == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    row["code"] = chunk["error"].get("code")
                    break
                if chunk.get("usage"):
                    row["completion_tokens"] = chunk["usage"]["completion_tokens"]
                for choice in chunk.get("choices", []):
                    if row["ttft"] is None and choice.get("delta", {}).get("content"):
                        row["ttft"] = time.perf_counter() - t0

            row["e2e"] = time.perf_counter() - t0
            if not done:
                row["code"] = row["code"] or "no_done_marker"
                resp.failure(row["code"])
            else:
                resp.success()
                if row["ttft"] is not None:
                    self._fire("TTFT", row["ttft"])
                self._fire("E2E", row["e2e"], row["completion_tokens"])
            _record(row)
