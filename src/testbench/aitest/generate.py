"""让本地 Ollama 模型根据接口文档生成测试用例。

按测试类别分多次提问，每次只要少量用例：小模型一次输出太长容易写乱。
schema=True 时用 Ollama 的约束解码（format 传 JSON Schema），解码阶段就只能产生合法 JSON；
schema=False 只在提示词里要求 JSON，用来对比约束解码对格式合法率的影响。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from testbench.aitest.cases import CASES_SCHEMA

SPEC = (Path(__file__).parent / "api_spec.md").read_text(encoding="utf-8")

CATEGORIES = {
    "auth": "鉴权：覆盖每个需要鉴权的接口在不带 Key、Key 无效、Key 有效时的表现，以及受限 Key 的权限。",
    "params": "对话接口的请求体校验：非法 JSON、缺少必填字段、类型错误、取值越界（temperature、top_p、max_tokens、n、role、messages）。",
    "models": "对话接口的模型检查与内容检查：模型不存在、Key 无权访问、模型类型不匹配、图片输入、拦截词。注意文档里的检查顺序。",
    "success": "对话接口的正常场景与边界值：普通请求、流式请求、取值恰好在边界上的合法参数、视觉模型的图片输入。",
    "embeddings": "Embeddings 接口：正常请求、各种非法 input、模型类型不匹配、鉴权。",
}

EXAMPLE = {
    "cases": [{
        "name": "temperature 超过上限",
        "method": "POST",
        "path": "/v1/chat/completions",
        "auth": "valid",
        "body": {"model": "qwen-mini", "messages": [{"role": "user", "content": "你好"}], "temperature": 3},
        "raw_body": None,
        "expect_status": 400,
        "expect_code": "invalid_parameter",
    }]
}

SYSTEM = "你是一名资深接口测试工程师，只根据给定的接口文档设计测试用例，不臆测文档以外的行为。"


def build_prompt(category: str, n: int) -> str:
    return f"""下面是接口文档：

<文档>
{SPEC}
</文档>

请设计 {n} 条测试用例，范围：{CATEGORIES[category]}

输出要求：
- 只输出一个 JSON 对象，形如 {{"cases": [...]}}，不要输出任何解释
- 每条用例的字段：name（中文描述）、method（GET 或 POST）、path、auth（valid / limited / invalid / missing 之一）、
  body（请求体对象，GET 请求为 null）、raw_body（只在测试非法 JSON 时填原始字符串，否则为 null）、
  expect_status（期望的 HTTP 状态码）、expect_code（期望的错误码，成功时为 null）
- 期望结果必须严格按文档推导，尤其注意文档里的检查顺序

示例（只是格式示例，不要照抄）：
{json.dumps(EXAMPLE, ensure_ascii=False, indent=1)}
"""


@dataclass
class Generation:
    model: str
    category: str
    schema: bool
    content: str
    seconds: float
    prompt_tokens: int
    output_tokens: int


def generate(model: str, category: str, schema: bool, n: int = 8, ollama: str = "http://localhost:11434",
             cache_dir: Path | None = None) -> Generation:
    cache = None
    if cache_dir:
        cache = cache_dir / f"{model.replace(':', '_')}-{'schema' if schema else 'free'}-{category}.json"
        if cache.exists():
            return Generation(**json.loads(cache.read_text(encoding="utf-8")))
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(category, n)}],
        "stream": False,
        "options": {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 3000},
    }
    if schema:
        body["format"] = CASES_SCHEMA
    t0 = time.monotonic()
    with httpx.Client(trust_env=False, timeout=900) as c:
        r = c.post(f"{ollama}/api/chat", json=body)
        r.raise_for_status()
        data = r.json()
    g = Generation(model, category, schema, data["message"]["content"], time.monotonic() - t0,
                   data.get("prompt_eval_count", 0), data.get("eval_count", 0))
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(g.__dict__, ensure_ascii=False, indent=1), encoding="utf-8")
    return g
