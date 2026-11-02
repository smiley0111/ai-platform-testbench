"""执行用例：按接口文档描述的环境起网关（Mock 后端），逐条发请求，比对状态码和错误码。

也可以命令行运行，供变异测试在子进程里对“被改坏的网关”执行同一批用例：
    python -m testbench.aitest.runner cases.json results.json
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import asdict, dataclass

from testbench.aitest.cases import Case, load_cases
from testbench.gateway import GatewayConfig, KeyPolicy, ModelSpec, create_app

KEYS = {"valid": "sk-valid", "limited": "sk-limited", "invalid": "sk-does-not-exist"}


def spec_config() -> GatewayConfig:
    """和 api_spec.md 描述一致的环境。限流调到很大：用例都是单个请求，不测限流。"""
    return GatewayConfig(
        backend="mock",
        keys={"sk-valid": KeyPolicy(rpm=1_000_000), "sk-limited": KeyPolicy(models={"qwen-mini"}, rpm=1_000_000)},
        models={
            "qwen-mini": ModelSpec("mock-mini", fallback="qwen-backup"),
            "qwen-backup": ModelSpec("mock-backup"),
            "qwen-vl": ModelSpec("mock-vl", vision=True),
            "embed-mini": ModelSpec("mock-embed", kind="embedding"),
        },
        blocked_words=["违禁词示例"],
    )


@dataclass
class Result:
    name: str
    passed: bool
    status: int
    code: str | None
    param: str | None
    expect_status: int
    expect_code: str | None
    stream: bool = False


def run_cases(cases: list[Case]) -> list[Result]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # starlette 对 httpx 的弃用提示，与测试无关
        from fastapi.testclient import TestClient

    client = TestClient(create_app(spec_config()), raise_server_exceptions=False)
    results = []
    for c in cases:
        headers = {} if c.auth == "missing" else {"Authorization": f"Bearer {KEYS[c.auth]}"}
        kwargs: dict = {"headers": headers}
        if c.raw_body is not None:
            kwargs["content"] = c.raw_body.encode()
            headers["Content-Type"] = "application/json"
        elif c.body is not None:
            kwargs["json"] = c.body
        r = client.request(c.method, c.path, **kwargs)
        code = param = None
        if r.status_code >= 400:
            try:
                err = r.json()["error"]
                code, param = err.get("code"), err.get("param")
            except (ValueError, KeyError, TypeError):
                code = "unparseable"
        stream = r.headers.get("content-type", "").startswith("text/event-stream")
        passed = r.status_code == c.expect_status and (c.expect_code is None or c.expect_code == code)
        if passed and r.status_code == 200 and (c.body or {}).get("stream") is True:
            passed = stream  # 文档约定：stream=true 必须以 SSE 返回
        results.append(Result(c.name, passed, r.status_code, code, param, c.expect_status, c.expect_code, stream))
    return results


def main() -> None:
    cases_path, out_path = sys.argv[1], sys.argv[2]
    results = run_cases(load_cases(cases_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False)


if __name__ == "__main__":
    main()
