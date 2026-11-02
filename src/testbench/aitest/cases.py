"""测试用例的数据格式：模型生成 JSON 数据而不是测试代码，执行器统一负责发请求和比对。

这样能把两类问题分开统计：格式不合法（模型没按要求输出）和期望写错（模型理解错了接口）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

AUTH = ("valid", "limited", "invalid", "missing")
METHODS = ("GET", "POST")
PATHS = ("/healthz", "/v1/models", "/v1/chat/completions", "/v1/embeddings")

# 约束解码用的 JSON Schema（Ollama 的 format 参数）
CASES_SCHEMA = {
    "type": "object",
    "properties": {
        "cases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "method": {"type": "string", "enum": list(METHODS)},
                    "path": {"type": "string", "enum": list(PATHS)},
                    "auth": {"type": "string", "enum": list(AUTH)},
                    "body": {"type": ["object", "null"]},
                    "raw_body": {"type": ["string", "null"]},
                    "expect_status": {"type": "integer"},
                    "expect_code": {"type": ["string", "null"]},
                },
                # 可空字段也列为必填：约束解码下，非必填字段会被模型直接省略（实测 body 全部丢失）
                "required": ["name", "method", "path", "auth", "body", "raw_body", "expect_status", "expect_code"],
            },
        }
    },
    "required": ["cases"],
}


@dataclass
class Case:
    name: str
    method: str
    path: str
    auth: str
    expect_status: int
    body: dict | None = None
    raw_body: str | None = None  # 用来测非法 JSON，优先于 body
    expect_code: str | None = None
    source: str = ""  # 来自哪次生成，便于追溯

    def key(self) -> str:
        """去重用：请求完全相同的用例只算一个。"""
        return json.dumps([self.method, self.path, self.auth, self.body, self.raw_body], sort_keys=True,
                          ensure_ascii=False)

    def to_dict(self) -> dict:
        return asdict(self)


class Malformed(ValueError):
    pass


def parse_case(obj: object, source: str = "") -> Case:
    """单条用例的格式校验。不合法就抛 Malformed，原因写清楚，报告里要统计。"""
    if not isinstance(obj, dict):
        raise Malformed("case is not an object")
    for f in ("name", "method", "path", "auth", "expect_status"):
        if f not in obj:
            raise Malformed(f"missing field {f}")
    if obj["method"] not in METHODS:
        raise Malformed(f"bad method {obj['method']!r}")
    if obj["path"] not in PATHS:
        raise Malformed(f"unknown path {obj['path']!r}")
    if obj["auth"] not in AUTH:
        raise Malformed(f"bad auth {obj['auth']!r}")
    status = obj["expect_status"]
    if not isinstance(status, int) or isinstance(status, bool) or not 100 <= status <= 599:
        raise Malformed(f"bad expect_status {status!r}")
    body, raw = obj.get("body"), obj.get("raw_body")
    if body is not None and not isinstance(body, dict):
        raise Malformed("body must be an object or null")
    if raw is not None and not isinstance(raw, str):
        raise Malformed("raw_body must be a string or null")
    code = obj.get("expect_code")
    if code is not None and not isinstance(code, str):
        raise Malformed("expect_code must be a string or null")
    if obj["method"] == "GET" and (body or raw):
        raise Malformed("GET request with a body")
    if obj["method"] == "POST" and body is None and raw is None:
        raise Malformed("POST request without body (use raw_body \"\" to test an empty body)")
    return Case(name=str(obj["name"]), method=obj["method"], path=obj["path"], auth=obj["auth"],
                expect_status=status, body=body, raw_body=raw, expect_code=code or None, source=source)


def parse_output(text: str, source: str = "") -> tuple[list[Case], list[str]]:
    """解析一次模型输出。返回（合法用例，每条不合法用例的原因）。整段不是 JSON 时整段算一条失败。"""
    try:
        data = json.loads(_strip_fence(text))
    except json.JSONDecodeError as e:
        return [], [f"output is not JSON: {e.msg}"]
    items = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return [], ["no 'cases' array"]
    cases, errors = [], []
    for item in items:
        try:
            cases.append(parse_case(item, source))
        except Malformed as e:
            errors.append(str(e))
    return cases, errors


def _strip_fence(text: str) -> str:
    """不带约束解码时，模型常把 JSON 包在 ```json 代码块里。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    return t


def load_cases(path) -> list[Case]:
    with open(path, encoding="utf-8") as f:
        return [Case(**c) for c in json.load(f)]


def save_cases(cases: list[Case], path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in cases], f, ensure_ascii=False, indent=1)
