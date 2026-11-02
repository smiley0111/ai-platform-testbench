"""从接口文档拆出的需求点，用来算需求覆盖率。

一条需求被覆盖 = 至少有一条“有效用例”（期望写对、实际执行通过）命中它。
期望写错的用例即使碰巧走到了某个分支也不算，因为它在真实回归里只会制造误报。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from testbench.aitest.cases import Case
from testbench.aitest.runner import Result

CHAT, EMB, MODELS, HEALTH = "/v1/chat/completions", "/v1/embeddings", "/v1/models", "/healthz"


@dataclass
class Requirement:
    id: str
    text: str
    match: Callable[[Case, Result], bool]


def _err(path: str, status: int, code: str, param: str | None = None):
    def m(c: Case, r: Result) -> bool:
        return (c.path == path and r.status == status and r.code == code
                and (param is None or (r.param or "").split(".")[0] == param
                     or (param == "role" and "role" in (r.param or ""))))
    return m


def _ok(path: str, pred: Callable[[Case], bool] = lambda c: True):
    return lambda c, r: c.path == path and r.status == 200 and pred(c)


def _body(c: Case) -> dict:
    return c.body or {}


def _has_image(c: Case) -> bool:
    for m in _body(c).get("messages") or []:
        if isinstance(m, dict) and isinstance(m.get("content"), list):
            if any(isinstance(p, dict) and p.get("type") == "image_url" for p in m["content"]):
                return True
    return False


def _boundary(c: Case) -> bool:
    b = _body(c)
    return b.get("temperature") in (0, 2) or b.get("top_p") in (0, 1) or b.get("max_tokens") in (1, 4096)


REQUIREMENTS: list[Requirement] = [
    Requirement("R01", "healthz 无需鉴权返回 200", _ok(HEALTH)),
    Requirement("R02", "模型列表返回 200", _ok(MODELS)),
    Requirement("R03", "模型列表缺 Key 返回 401", _err(MODELS, 401, "missing_api_key")),
    Requirement("R04", "对话非流式成功", _ok(CHAT, lambda c: not _body(c).get("stream"))),
    Requirement("R05", "对话流式成功（SSE）", lambda c, r: _ok(CHAT, lambda c: _body(c).get("stream") is True)(c, r) and r.stream),
    Requirement("R06", "对话缺 Key 401", _err(CHAT, 401, "missing_api_key")),
    Requirement("R07", "对话 Key 无效 401", _err(CHAT, 401, "invalid_api_key")),
    Requirement("R08", "请求体不是合法 JSON 400", _err(CHAT, 400, "invalid_json")),
    Requirement("R09", "messages 缺失或为空 400", _err(CHAT, 400, "invalid_parameter", "messages")),
    Requirement("R10", "model 缺失 400", _err(CHAT, 400, "invalid_parameter", "model")),
    Requirement("R11", "temperature 越界 400", _err(CHAT, 400, "invalid_parameter", "temperature")),
    Requirement("R12", "top_p 越界 400", _err(CHAT, 400, "invalid_parameter", "top_p")),
    Requirement("R13", "max_tokens 越界 400", _err(CHAT, 400, "invalid_parameter", "max_tokens")),
    Requirement("R14", "n 不为 1 时 400", _err(CHAT, 400, "invalid_parameter", "n")),
    Requirement("R15", "非法 role 400", _err(CHAT, 400, "invalid_parameter", "role")),
    Requirement("R16", "模型不存在 404", _err(CHAT, 404, "model_not_found")),
    Requirement("R17", "Key 无权访问模型 403", _err(CHAT, 403, "model_not_allowed")),
    Requirement("R18", "对话接口用 embedding 模型 400", _err(CHAT, 400, "model_kind_mismatch")),
    Requirement("R19", "不支持图片的模型收到图片 400", _err(CHAT, 400, "image_not_supported")),
    Requirement("R20", "输入命中拦截词 400", _err(CHAT, 400, "content_filter")),
    Requirement("R21", "取值边界内的参数被接受（如 temperature=2、max_tokens=4096）", _ok(CHAT, _boundary)),
    Requirement("R22", "视觉模型接受图片输入", _ok(CHAT, _has_image)),
    Requirement("R23", "Embeddings 成功", _ok(EMB)),
    Requirement("R24", "Embeddings input 为空 400", _err(EMB, 400, "invalid_parameter", "input")),
    Requirement("R25", "Embeddings 用对话模型 400", _err(EMB, 400, "model_kind_mismatch")),
    Requirement("R26", "Embeddings 缺 Key 401", _err(EMB, 401, "missing_api_key")),
]


def coverage(cases: list[Case], results: list[Result]) -> dict[str, bool]:
    covered = {req.id: False for req in REQUIREMENTS}
    for c, r in zip(cases, results, strict=True):
        if not r.passed:
            continue
        for req in REQUIREMENTS:
            if not covered[req.id] and req.match(c, r):
                covered[req.id] = True
    return covered
