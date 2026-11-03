"""推理后端：Ollama（真实模型）和 Mock（可注入故障，用于可重复的可靠性测试）。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from testbench.gateway.schemas import ChatCompletionRequest


class BackendError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class Fault:
    mode: str = "ok"  # ok | error | flaky | slow | stream_break
    status: int = 500
    delay: float = 0.0
    fail_times: int = 0
    reply: str | None = None  # 强制模型输出的内容，用于测试输出过滤


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


class MockBackend:
    """回显式假模型：回复 `mock reply: <最后一条用户消息>`，按空格切分 token。"""

    def __init__(self, latency: float = 0.0):
        self.faults: dict[str, Fault] = {}
        self.calls: dict[str, int] = {}
        self.latency = latency  # 每次调用固定耗时，模拟推理时间

    def set_fault(self, model: str, fault: Fault) -> None:
        self.faults[model] = fault
        self.calls[model] = 0

    def reset(self) -> None:
        self.faults.clear()
        self.calls.clear()

    async def _apply_fault(self, model: str) -> Fault:
        fault = self.faults.get(model) or self.faults.get("*") or Fault()
        n = self.calls.get(model, 0) + 1
        self.calls[model] = n
        if fault.mode == "error":
            raise BackendError(fault.status, f"injected error {fault.status}")
        if fault.mode == "flaky" and n <= fault.fail_times:
            raise BackendError(fault.status, f"injected flaky error {n}/{fault.fail_times}")
        if fault.mode == "slow":
            await asyncio.sleep(fault.delay)
        if self.latency:
            await asyncio.sleep(self.latency)
        return fault

    @staticmethod
    def _tokens(req: ChatCompletionRequest, fault: Fault) -> tuple[list[str], str]:
        last_user = next((m.text() for m in reversed(req.messages) if m.role == "user"), "")
        reply = fault.reply if fault.reply is not None else f"mock reply: {last_user}"
        finish = "stop"
        stops = [req.stop] if isinstance(req.stop, str) else (req.stop or [])
        for s in stops:
            if s and s in reply:
                reply = reply[: reply.index(s)]
        tokens = reply.split(" ")
        if req.max_tokens is not None and len(tokens) > req.max_tokens:
            tokens, finish = tokens[: req.max_tokens], "length"
        return tokens, finish

    @staticmethod
    def _prompt_tokens(req: ChatCompletionRequest) -> int:
        return sum(len(m.text().split(" ")) for m in req.messages)

    async def chat(self, model: str, backend_name: str, req: ChatCompletionRequest) -> dict:
        fault = await self._apply_fault(model)
        tokens, finish = self._tokens(req, fault)
        prompt = self._prompt_tokens(req)
        return {
            "id": _completion_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": " ".join(tokens)},
                    "finish_reason": finish,
                }
            ],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": len(tokens),
                "total_tokens": prompt + len(tokens),
            },
        }

    async def chat_stream(
        self, model: str, backend_name: str, req: ChatCompletionRequest
    ) -> AsyncIterator[dict]:
        fault = await self._apply_fault(model)
        tokens, finish = self._tokens(req, fault)
        cid, created = _completion_id(), int(time.time())

        def chunk(delta: dict, finish_reason: str | None = None) -> dict:
            return {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }

        yield chunk({"role": "assistant", "content": ""})
        for i, tok in enumerate(tokens):
            if fault.mode == "stream_break" and i == 2:
                raise BackendError(502, "injected stream break")
            yield chunk({"content": tok if i == 0 else " " + tok})
        yield chunk({}, finish)
        if req.stream_options and req.stream_options.include_usage:
            prompt = self._prompt_tokens(req)
            yield {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt,
                    "completion_tokens": len(tokens),
                    "total_tokens": prompt + len(tokens),
                },
            }

    async def embeddings(self, model: str, backend_name: str, inputs: list[str]) -> list[list[float]]:
        await self._apply_fault(model)
        vectors = []
        for text in inputs:
            digest = hashlib.sha256(text.encode()).digest()
            v = [b / 255 - 0.5 for b in digest[:8]]
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vectors.append([x / norm for x in v])
        return vectors


class OllamaBackend:
    """调用 Ollama 自带的 OpenAI 兼容接口（/v1/...）。超时由网关统一控制。"""

    def __init__(self, base_url: str):
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=None)

    def _payload(self, backend_name: str, req: ChatCompletionRequest) -> dict:
        body = req.model_dump(exclude_none=True)
        body["model"] = backend_name
        return body

    async def chat(self, model: str, backend_name: str, req: ChatCompletionRequest) -> dict:
        try:
            r = await self.client.post("/v1/chat/completions", json=self._payload(backend_name, req))
        except httpx.TransportError as e:
            raise BackendError(503, f"backend unreachable: {e}") from e
        if r.status_code >= 400:
            raise BackendError(r.status_code, r.text)
        data = r.json()
        data["model"] = model
        return data

    async def chat_stream(
        self, model: str, backend_name: str, req: ChatCompletionRequest
    ) -> AsyncIterator[dict]:
        try:
            async with self.client.stream(
                "POST", "/v1/chat/completions", json=self._payload(backend_name, req)
            ) as r:
                if r.status_code >= 400:
                    raise BackendError(r.status_code, (await r.aread()).decode())
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: ") :]
                    if data.strip() == "[DONE]":
                        return
                    chunk = json.loads(data)
                    chunk["model"] = model
                    yield chunk
        except httpx.TransportError as e:
            raise BackendError(503, f"backend unreachable: {e}") from e

    async def embeddings(self, model: str, backend_name: str, inputs: list[str]) -> list[list[float]]:
        try:
            r = await self.client.post(
                "/v1/embeddings", json={"model": backend_name, "input": inputs}
            )
        except httpx.TransportError as e:
            raise BackendError(503, f"backend unreachable: {e}") from e
        if r.status_code >= 400:
            raise BackendError(r.status_code, r.text)
        return [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
