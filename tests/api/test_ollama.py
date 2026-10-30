"""接真实模型（本地 Ollama）跑一遍主流程，验证网关和真实后端之间的兼容性。

Mock 能证明网关逻辑对，但证明不了真实后端返回的格式和 Mock 一样。
Ollama 没启动或模型没下载时自动跳过。
"""

import json
import os
import time

import httpx
import openai
import pytest
from conftest import read_sse, sse_json, start_server

from testbench.gateway import GatewayConfig, KeyPolicy, ModelSpec, create_app

pytestmark = pytest.mark.ollama

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("TESTBENCH_CHAT_MODEL", "qwen2.5:0.5b")
EMBED_MODEL = os.getenv("TESTBENCH_EMBED_MODEL", "nomic-embed-text")
KEY = "sk-ollama-test"


def _pulled_models() -> set[str]:
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2, trust_env=False)
        return {m["name"] for m in r.json().get("models", [])}
    except (httpx.HTTPError, ValueError):
        return set()


def _has(models: set[str], name: str) -> bool:
    return name in models or f"{name}:latest" in models


@pytest.fixture(scope="module")
def ollama_url():
    models = _pulled_models()
    if not _has(models, CHAT_MODEL):
        pytest.skip(f"Ollama 不可用或未下载 {CHAT_MODEL}（ollama pull {CHAT_MODEL}）")
    config = GatewayConfig(
        backend="ollama", ollama_url=OLLAMA_URL, keys={KEY: KeyPolicy()},
        models={"qwen-mini": ModelSpec(CHAT_MODEL),
                "embed-mini": ModelSpec(EMBED_MODEL, kind="embedding")},
        request_timeout=120,
    )
    url, server, thread = start_server(create_app(config))
    yield url, models
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def http(ollama_url):
    url, _ = ollama_url
    with httpx.Client(base_url=url, headers={"authorization": f"Bearer {KEY}"},
                      timeout=120, trust_env=False) as c:
        yield c


def body(**kwargs):
    return {"model": "qwen-mini", "messages": [{"role": "user", "content": "用一句话介绍青岛"}],
            "max_tokens": 32, "temperature": 0, **kwargs}


def test_chat(http):
    data = http.post("/v1/chat/completions", json=body()).json()
    assert data["choices"][0]["message"]["content"].strip()
    assert data["model"] == "qwen-mini", "对外只暴露网关的模型名，不暴露后端真实模型名"
    assert data["usage"]["completion_tokens"] <= 32


def test_stream_and_time_to_first_token(http):
    start = time.monotonic()
    first_token_at = None
    with http.stream("POST", "/v1/chat/completions", json=body(stream=True)) as r:
        assert r.status_code == 200
        events = []
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            events.append(line[6:])
            if first_token_at is None and events[-1] != "[DONE]":
                choices = json.loads(events[-1]).get("choices") or [{}]
                if choices[0].get("delta", {}).get("content"):
                    first_token_at = time.monotonic()
    assert events[-1] == "[DONE]"
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in sse_json(events) if c["choices"])
    assert text.strip()
    assert first_token_at is not None
    print(f"\n首字延迟 {first_token_at - start:.2f}s，总耗时 {time.monotonic() - start:.2f}s")


def test_stream_equals_non_stream_at_temperature_zero(http):
    """temperature=0 时同一个输入，流式和非流式结果应一致（不同推理引擎不一定保证，这里记录现象）。"""
    expected = http.post("/v1/chat/completions", json=body()).json()["choices"][0]["message"]["content"]
    with http.stream("POST", "/v1/chat/completions", json=body(stream=True)) as r:
        chunks = sse_json(read_sse(r))
    streamed = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
    assert streamed == expected


def test_openai_sdk(ollama_url):
    url, _ = ollama_url
    client = openai.OpenAI(base_url=f"{url}/v1", api_key=KEY, max_retries=0,
                           http_client=httpx.Client(trust_env=False, timeout=120))
    resp = client.chat.completions.create(**body())
    assert resp.choices[0].message.content.strip()


def test_embeddings(http, ollama_url):
    _, models = ollama_url
    if not _has(models, EMBED_MODEL):
        pytest.skip(f"未下载 {EMBED_MODEL}（ollama pull {EMBED_MODEL}）")
    data = http.post("/v1/embeddings", json={"model": "embed-mini", "input": ["青岛", "啤酒"]}).json()
    assert len(data["data"]) == 2
    assert len(data["data"][0]["embedding"]) == len(data["data"][1]["embedding"]) > 0
