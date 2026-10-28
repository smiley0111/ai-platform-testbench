"""OpenAI 兼容推理网关：鉴权 → 限流 → 参数校验 → 内容过滤 → 重试 / 超时 / 熔断 / 降级 → 后端。"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from testbench.gateway.backends import BackendError, Fault, MockBackend, OllamaBackend
from testbench.gateway.config import GatewayConfig, KeyPolicy, ModelSpec
from testbench.gateway.resilience import CircuitBreaker, TokenBucket, retry_after_header
from testbench.gateway.schemas import ChatCompletionRequest, EmbeddingRequest


class APIError(Exception):
    def __init__(
        self,
        status: int,
        message: str,
        type_: str,
        code: str | None = None,
        param: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(message)
        self.status, self.message, self.type, self.code, self.param = status, message, type_, code, param
        self.headers = headers or {}

    def body(self) -> dict:
        return {"error": {"message": self.message, "type": self.type, "param": self.param, "code": self.code}}


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(config: GatewayConfig | None = None, backend: Any = None) -> FastAPI:
    config = config or GatewayConfig.default()
    if backend is None:
        backend = MockBackend() if config.backend == "mock" else OllamaBackend(config.ollama_url)

    app = FastAPI(title="ai-platform-testbench gateway")
    state: dict[str, Any] = {}

    def reset_state() -> None:
        state["buckets"] = {k: TokenBucket(p.rpm) for k, p in config.keys.items()}
        state["breakers"] = {
            m: CircuitBreaker(config.breaker_failures, config.breaker_reset) for m in config.models
        }

    reset_state()
    app.state.config, app.state.backend, app.state.gateway = config, backend, state

    @app.exception_handler(APIError)
    async def _api_error(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(exc.body(), status_code=exc.status, headers=exc.headers)

    @app.middleware("http")
    async def _request_id(request: Request, call_next):
        rid = request.headers.get("x-request-id") or f"req-{uuid.uuid4().hex[:16]}"
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response

    # ---------- 公共检查 ----------

    def authenticate(request: Request) -> tuple[str, KeyPolicy]:
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer ") or not auth[7:].strip():
            raise APIError(401, "Missing API key. Use header 'Authorization: Bearer <key>'.",
                           "authentication_error", "missing_api_key")
        key = auth[7:].strip()
        policy = config.keys.get(key)
        if policy is None:
            raise APIError(401, "Invalid API key.", "authentication_error", "invalid_api_key")
        wait = state["buckets"][key].acquire()
        if wait > 0:
            raise APIError(429, f"Rate limit exceeded: {policy.rpm} requests per minute.",
                           "rate_limit_error", "rate_limit_exceeded",
                           headers={"retry-after": retry_after_header(wait)})
        return key, policy

    def resolve_model(name: str, policy: KeyPolicy, kind: str) -> ModelSpec:
        spec = config.models.get(name)
        if spec is None:
            raise APIError(404, f"The model '{name}' does not exist.", "invalid_request_error",
                           "model_not_found", "model")
        if policy.models is not None and name not in policy.models:
            raise APIError(403, f"This API key is not allowed to use model '{name}'.",
                           "permission_error", "model_not_allowed", "model")
        if spec.kind != kind:
            raise APIError(400, f"Model '{name}' is a {spec.kind} model and does not support this endpoint.",
                           "invalid_request_error", "model_kind_mismatch", "model")
        return spec

    async def parse(request: Request, schema):
        try:
            raw = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise APIError(400, "Request body is not valid JSON.", "invalid_request_error", "invalid_json") from e
        try:
            return schema.model_validate(raw)
        except ValidationError as e:
            err = e.errors()[0]
            param = ".".join(str(p) for p in err["loc"]) or None
            raise APIError(400, f"Invalid parameter '{param}': {err['msg']}", "invalid_request_error",
                           "invalid_parameter", param) from e

    def blocked(text: str) -> bool:
        return any(w in text for w in config.blocked_words)

    def mask(text: str) -> str:
        for w in config.blocked_words:
            text = text.replace(w, "*" * len(w))
        return text

    # ---------- 重试 / 超时 / 熔断 / 降级 ----------

    async def with_retry(fn: Callable[[], Awaitable[Any]]) -> Any:
        for attempt in range(config.max_retries + 1):
            try:
                return await asyncio.wait_for(fn(), config.request_timeout)
            except asyncio.TimeoutError as e:
                raise APIError(504, f"Upstream model did not respond within {config.request_timeout}s.",
                               "timeout_error", "upstream_timeout") from e
            except BackendError as e:
                if e.status < 500 and e.status != 429:
                    raise APIError(e.status, e.message, "invalid_request_error", "upstream_rejected") from e
                if attempt == config.max_retries:
                    raise APIError(502, f"Upstream error after {attempt + 1} attempts: {e.message}",
                                   "upstream_error", "upstream_error") from e
                await asyncio.sleep(config.retry_backoff * 2**attempt)

    async def with_fallback(public: str, call: Callable[[str], Awaitable[Any]]) -> tuple[Any, str]:
        chain = [public]
        if config.models[public].fallback:
            chain.append(config.models[public].fallback)
        last: APIError | None = None
        for name in chain:
            breaker = state["breakers"][name]
            if not breaker.allow():
                last = APIError(503, f"Model '{name}' is temporarily unavailable (circuit open).",
                                "service_unavailable", "circuit_open")
                continue
            try:
                result = await with_retry(lambda n=name: call(n))
            except APIError as e:
                if e.status < 500:
                    breaker.record_success()
                    raise
                breaker.record_failure()
                last = e
                continue
            breaker.record_success()
            return result, name
        assert last is not None
        raise last

    def served_headers(public: str, served: str) -> dict[str, str]:
        return {"x-served-model": served, "x-fallback": "true" if served != public else "false"}

    # ---------- 接口 ----------

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "backend": config.backend}

    @app.get("/v1/models")
    async def list_models(request: Request) -> dict:
        _, policy = authenticate(request)
        names = [m for m in config.models if policy.models is None or m in policy.models]
        return {"object": "list",
                "data": [{"id": m, "object": "model", "created": 0, "owned_by": "testbench"} for m in names]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        _, policy = authenticate(request)
        req: ChatCompletionRequest = await parse(request, ChatCompletionRequest)
        spec = resolve_model(req.model, policy, "chat")
        if req.max_tokens is not None and req.max_tokens > config.max_tokens_limit:
            raise APIError(400, f"max_tokens must be <= {config.max_tokens_limit}.",
                           "invalid_request_error", "invalid_parameter", "max_tokens")
        if any(m.has_image() for m in req.messages) and not spec.vision:
            raise APIError(400, f"Model '{req.model}' does not support image input.",
                           "invalid_request_error", "image_not_supported", "messages")
        if any(blocked(m.text()) for m in req.messages):
            raise APIError(400, "Input was rejected by the content filter.",
                           "invalid_request_error", "content_filter", "messages")

        def backend_name(n: str) -> str:
            return config.models[n].backend_name

        if not req.stream:
            data, served = await with_fallback(
                req.model, lambda n: backend.chat(n, backend_name(n), req))
            for choice in data.get("choices", []):
                content = choice.get("message", {}).get("content") or ""
                if blocked(content):
                    choice["message"]["content"] = mask(content)
                    choice["finish_reason"] = "content_filter"
            return JSONResponse(data, headers=served_headers(req.model, served))

        async def open_stream(n: str):
            it = backend.chat_stream(n, backend_name(n), req).__aiter__()
            return it, await it.__anext__()

        (it, first), served = await with_fallback(req.model, open_stream)

        async def events() -> AsyncIterator[str]:
            yield _sse(first)
            try:
                async for chunk in it:
                    for choice in chunk.get("choices", []):
                        content = choice.get("delta", {}).get("content")
                        if content and blocked(content):
                            choice["delta"]["content"] = mask(content)
                            choice["finish_reason"] = "content_filter"
                    yield _sse(chunk)
            except BackendError as e:
                state["breakers"][served].record_failure()
                yield _sse(APIError(502, f"Stream interrupted: {e.message}", "upstream_error",
                                    "stream_interrupted").body())
                return
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"cache-control": "no-cache", **served_headers(req.model, served)})

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> JSONResponse:
        _, policy = authenticate(request)
        req: EmbeddingRequest = await parse(request, EmbeddingRequest)
        spec = resolve_model(req.model, policy, "embedding")
        texts = req.texts()
        vectors, served = await with_fallback(
            req.model, lambda n: backend.embeddings(n, spec.backend_name, texts))
        tokens = sum(len(t.split(" ")) for t in texts)
        return JSONResponse({
            "object": "list",
            "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)],
            "model": req.model,
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        }, headers=served_headers(req.model, served))

    # ---------- 故障注入（只在 Mock 后端且配置了 admin_token 时开放）----------

    if config.admin_token and isinstance(backend, MockBackend):

        def check_admin(request: Request) -> None:
            if request.headers.get("x-admin-token") != config.admin_token:
                raise APIError(403, "Invalid admin token.", "permission_error", "invalid_admin_token")

        @app.post("/admin/faults")
        async def set_fault(request: Request) -> dict:
            check_admin(request)
            body = await request.json()
            model = body.pop("model", "*")
            backend.set_fault(model, Fault(**body))
            return {"ok": True, "model": model}

        @app.post("/admin/reset")
        async def reset(request: Request) -> dict:
            check_admin(request)
            backend.reset()
            reset_state()
            return {"ok": True}

        @app.get("/admin/state")
        async def get_state(request: Request) -> dict:
            check_admin(request)
            return {"breakers": {m: b.state for m, b in state["breakers"].items()},
                    "calls": dict(backend.calls)}

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
