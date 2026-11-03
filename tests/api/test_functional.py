"""功能正确性：响应结构、参数校验、错误码、多模态输入、Embeddings。"""

import pytest
from apikit import KEY_LIMITED, assert_error, chat_body


class TestChatResponse:
    def test_response_schema(self, client):
        r = client.post("/v1/chat/completions", json=chat_body("hello world"))
        assert r.status_code == 200
        data = r.json()
        assert data["id"].startswith("chatcmpl-")
        assert data["object"] == "chat.completion"
        assert isinstance(data["created"], int)
        assert data["model"] == "qwen-mini"
        choice = data["choices"][0]
        assert choice["index"] == 0
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"]
        assert choice["finish_reason"] == "stop"

    def test_usage_adds_up(self, client):
        usage = client.post("/v1/chat/completions", json=chat_body()).json()["usage"]
        assert usage["prompt_tokens"] > 0 and usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_multi_turn_answers_last_user_message(self, client):
        body = {"model": "qwen-mini", "messages": [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ]}
        content = client.post("/v1/chat/completions", json=body).json()["choices"][0]["message"]["content"]
        assert "second question" in content

    def test_max_tokens_truncates_with_length_reason(self, client):
        data = client.post("/v1/chat/completions",
                           json=chat_body("one two three four five six", max_tokens=3)).json()
        assert data["choices"][0]["finish_reason"] == "length"
        assert data["usage"]["completion_tokens"] == 3

    def test_stop_sequence(self, client):
        data = client.post("/v1/chat/completions",
                           json=chat_body("alpha beta STOP gamma", stop=["STOP"])).json()
        content = data["choices"][0]["message"]["content"]
        assert "STOP" not in content and "gamma" not in content

    def test_request_id_generated_and_echoed(self, client):
        assert client.post("/v1/chat/completions", json=chat_body()).headers["x-request-id"]
        r = client.post("/v1/chat/completions", json=chat_body(), headers={"x-request-id": "trace-123"})
        assert r.headers["x-request-id"] == "trace-123"


class TestParameterValidation:
    @pytest.mark.parametrize("body, param", [
        ({"messages": [{"role": "user", "content": "hi"}]}, "model"),
        ({"model": "qwen-mini"}, "messages"),
        ({"model": "qwen-mini", "messages": []}, "messages"),
        ({"model": "qwen-mini", "messages": [{"role": "robot", "content": "hi"}]}, "role"),
        ({"model": "qwen-mini", "messages": [{"role": "user", "content": 123}]}, "content"),
        (chat_body(temperature=2.5), "temperature"),
        (chat_body(temperature=-0.1), "temperature"),
        (chat_body(top_p=1.5), "top_p"),
        (chat_body(max_tokens=0), "max_tokens"),
        (chat_body(n=2), "n"),
        (chat_body(stream="maybe"), "stream"),
    ], ids=["no-model", "no-messages", "empty-messages", "bad-role", "bad-content-type",
            "temperature-high", "temperature-negative", "top_p-high", "max_tokens-zero",
            "n-unsupported", "stream-not-bool"])
    def test_invalid_parameters_return_400(self, client, body, param):
        err = assert_error(client.post("/v1/chat/completions", json=body), 400,
                           "invalid_parameter", param)
        assert err["type"] == "invalid_request_error"

    @pytest.mark.parametrize("temperature", [0, 2])
    def test_temperature_boundaries_accepted(self, client, temperature):
        r = client.post("/v1/chat/completions", json=chat_body(temperature=temperature))
        assert r.status_code == 200

    def test_max_tokens_over_limit(self, client):
        assert_error(client.post("/v1/chat/completions", json=chat_body(max_tokens=100_000)),
                     400, "invalid_parameter", "max_tokens")

    def test_invalid_json(self, client):
        r = client.post("/v1/chat/completions", content=b"{not json",
                        headers={"content-type": "application/json"})
        assert_error(r, 400, "invalid_json")

    def test_unknown_model_returns_404(self, client):
        assert_error(client.post("/v1/chat/completions", json=chat_body(model="gpt-99")),
                     404, "model_not_found", "model")

    def test_embedding_model_on_chat_endpoint(self, client):
        assert_error(client.post("/v1/chat/completions", json=chat_body(model="embed-mini")),
                     400, "model_kind_mismatch")


class TestMultimodal:
    @staticmethod
    def image_message(model: str) -> dict:
        return {"model": model, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what toy is this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
        ]}]}

    def test_vision_model_accepts_image(self, client):
        r = client.post("/v1/chat/completions", json=self.image_message("qwen-vl"))
        assert r.status_code == 200
        assert "what toy is this" in r.json()["choices"][0]["message"]["content"]

    def test_text_model_rejects_image(self, client):
        assert_error(client.post("/v1/chat/completions", json=self.image_message("qwen-mini")),
                     400, "image_not_supported")

    def test_unknown_content_part_type(self, client):
        body = {"model": "qwen-vl", "messages": [{"role": "user", "content": [
            {"type": "audio", "data": "..."}]}]}
        assert_error(client.post("/v1/chat/completions", json=body), 400, "invalid_parameter")


class TestModels:
    def test_list_models(self, client):
        data = client.get("/v1/models").json()
        assert data["object"] == "list"
        assert {"qwen-mini", "embed-mini"} <= {m["id"] for m in data["data"]}

    def test_list_models_respects_key_permission(self, client):
        r = client.get("/v1/models", headers={"authorization": f"Bearer {KEY_LIMITED}"})
        assert [m["id"] for m in r.json()["data"]] == ["qwen-mini"]


class TestEmbeddings:
    def test_single_input(self, client):
        data = client.post("/v1/embeddings", json={"model": "embed-mini", "input": "hello"}).json()
        assert data["object"] == "list" and data["model"] == "embed-mini"
        assert len(data["data"]) == 1 and data["data"][0]["object"] == "embedding"
        assert all(isinstance(x, float) for x in data["data"][0]["embedding"])

    def test_batch_keeps_order_and_dimension(self, client):
        texts = ["a", "b", "c"]
        data = client.post("/v1/embeddings", json={"model": "embed-mini", "input": texts}).json()["data"]
        assert [d["index"] for d in data] == [0, 1, 2]
        assert len({len(d["embedding"]) for d in data}) == 1

    def test_same_input_same_vector(self, client):
        def embed():
            return client.post("/v1/embeddings", json={"model": "embed-mini", "input": "same"}).json()
        assert embed()["data"][0]["embedding"] == embed()["data"][0]["embedding"]

    @pytest.mark.parametrize("value", ["", [], [""]], ids=["empty-string", "empty-list", "list-with-empty"])
    def test_empty_input_rejected(self, client, value):
        assert_error(client.post("/v1/embeddings", json={"model": "embed-mini", "input": value}),
                     400, "invalid_parameter", "input")

    def test_chat_model_on_embeddings_endpoint(self, client):
        assert_error(client.post("/v1/embeddings", json={"model": "qwen-mini", "input": "x"}),
                     400, "model_kind_mismatch")
