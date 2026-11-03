"""安全性：API Key 鉴权、RBAC 模型权限、请求频率限制、内容过滤、管理接口保护。"""

import pytest
from apikit import KEY_ALL, KEY_LIMITED, KEY_RATE, assert_error, chat_body, read_sse, sse_json


def bearer(key: str) -> dict:
    return {"authorization": f"Bearer {key}"}


def post(anon, body, headers=None):
    return anon.post("/v1/chat/completions", json=body, headers=headers or {})


class TestAuthentication:
    @pytest.mark.parametrize("headers, code", [
        ({}, "missing_api_key"),
        ({"authorization": "Bearer"}, "missing_api_key"),
        ({"authorization": f"Token {KEY_ALL}"}, "missing_api_key"),
        (bearer("sk-wrong"), "invalid_api_key"),
        (bearer(KEY_ALL.upper()), "invalid_api_key"),
    ], ids=["no-header", "empty-bearer", "wrong-scheme", "wrong-key", "case-changed-key"])
    def test_rejected(self, anon, headers, code):
        err = assert_error(post(anon, chat_body(), headers), 401, code)
        assert err["type"] == "authentication_error"

    def test_auth_checked_before_body_validation(self, anon):
        assert_error(post(anon, {"bad": "body"}, bearer("sk-wrong")), 401, "invalid_api_key")

    def test_error_does_not_echo_the_key(self, anon):
        r = post(anon, chat_body(), bearer("sk-secret-should-not-leak"))
        assert "sk-secret-should-not-leak" not in r.text

    def test_models_endpoint_requires_key(self, anon):
        assert_error(anon.get("/v1/models"), 401, "missing_api_key")

    def test_embeddings_endpoint_requires_key(self, anon):
        r = anon.post("/v1/embeddings", json={"model": "embed-mini", "input": "x"})
        assert_error(r, 401, "missing_api_key")


class TestRBAC:
    def test_allowed_model(self, anon):
        assert post(anon, chat_body(model="qwen-mini"), bearer(KEY_LIMITED)).status_code == 200

    def test_forbidden_model(self, anon):
        assert_error(post(anon, chat_body(model="qwen-vl"), bearer(KEY_LIMITED)),
                     403, "model_not_allowed", "model")

    def test_forbidden_model_on_embeddings(self, anon):
        r = anon.post("/v1/embeddings", json={"model": "embed-mini", "input": "x"},
                      headers=bearer(KEY_LIMITED))
        assert_error(r, 403, "model_not_allowed")


class TestRateLimit:
    def test_limit_then_429_with_retry_after(self, anon):
        codes = [post(anon, chat_body(), bearer(KEY_RATE)).status_code for _ in range(3)]
        assert codes == [200] * 3
        r = post(anon, chat_body(), bearer(KEY_RATE))
        assert_error(r, 429, "rate_limit_exceeded")
        assert int(r.headers["retry-after"]) >= 1

    def test_limit_is_per_key(self, anon):
        for _ in range(4):
            post(anon, chat_body(), bearer(KEY_RATE))
        assert post(anon, chat_body(), bearer(KEY_ALL)).status_code == 200

    def test_invalid_requests_also_consume_quota(self, anon):
        """频率限制在参数校验之前，否则攻击者可以用错误请求无限刷接口。"""
        for _ in range(3):
            post(anon, {"bad": "body"}, bearer(KEY_RATE))
        assert_error(post(anon, chat_body(), bearer(KEY_RATE)), 429)


class TestContentFilter:
    @pytest.mark.parametrize("role", ["user", "system"])
    def test_blocked_input(self, client, role):
        body = {"model": "qwen-mini", "messages": [{"role": role, "content": "包含违禁词示例的内容"}]}
        assert_error(client.post("/v1/chat/completions", json=body), 400, "content_filter")

    def test_blocked_input_inside_multimodal_text(self, client):
        body = {"model": "qwen-vl", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "违禁词示例"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}]}
        assert_error(client.post("/v1/chat/completions", json=body), 400, "content_filter")

    def test_blocked_output_is_masked(self, client, admin):
        admin.fault("qwen-mini", reply="模型输出了 违禁词示例 这样的内容")
        choice = client.post("/v1/chat/completions", json=chat_body()).json()["choices"][0]
        assert "违禁词示例" not in choice["message"]["content"]
        assert choice["finish_reason"] == "content_filter"

    def test_blocked_output_is_masked_in_stream(self, client, admin):
        admin.fault("qwen-mini", reply="正常 违禁词示例 正常")
        with client.stream("POST", "/v1/chat/completions", json=chat_body(stream=True)) as r:
            chunks = sse_json(read_sse(r))
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
        assert "违禁词示例" not in text


class TestAdminEndpoints:
    @pytest.mark.parametrize("headers", [{}, {"x-admin-token": "wrong"}])
    def test_admin_requires_token(self, anon, headers):
        assert_error(anon.post("/admin/reset", headers=headers), 403, "invalid_admin_token")
