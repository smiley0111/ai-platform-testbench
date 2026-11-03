"""流式响应（SSE）：事件格式、内容一致性、usage、中途断流、首包前重试。"""

from apikit import chat_body, read_sse, sse_json


def stream(client, body):
    return client.stream("POST", "/v1/chat/completions", json={**body, "stream": True})


class TestStreamFormat:
    def test_sse_headers_and_done_marker(self, client):
        with stream(client, chat_body("a b c")) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            events = read_sse(r)
        assert events[-1] == "[DONE]"
        assert events.count("[DONE]") == 1

    def test_chunk_structure(self, client):
        with stream(client, chat_body("a b c")) as r:
            chunks = sse_json(read_sse(r))
        assert len({c["id"] for c in chunks}) == 1, "同一次响应的所有 chunk 应共用一个 id"
        assert all(c["object"] == "chat.completion.chunk" for c in chunks)
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        finishes = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]]
        assert [f for f in finishes if f] == ["stop"], "只有最后一个 chunk 带 finish_reason"

    def test_stream_content_equals_non_stream(self, client):
        body = chat_body("the quick brown fox")
        expected = client.post("/v1/chat/completions", json=body).json()["choices"][0]["message"]["content"]
        with stream(client, body) as r:
            chunks = sse_json(read_sse(r))
        streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
        assert streamed == expected

    def test_include_usage_adds_final_usage_chunk(self, client):
        body = chat_body("a b c", stream_options={"include_usage": True})
        with stream(client, body) as r:
            chunks = sse_json(read_sse(r))
        last = chunks[-1]
        assert last["choices"] == []
        assert last["usage"]["total_tokens"] == last["usage"]["prompt_tokens"] + last["usage"]["completion_tokens"]
        assert all("usage" not in c for c in chunks[:-1])

    def test_no_usage_chunk_by_default(self, client):
        with stream(client, chat_body("a b c")) as r:
            chunks = sse_json(read_sse(r))
        assert all("usage" not in c for c in chunks)

    def test_stream_max_tokens(self, client):
        with stream(client, chat_body("one two three four five", max_tokens=2)) as r:
            chunks = sse_json(read_sse(r))
        assert chunks[-1]["choices"][0]["finish_reason"] == "length"


class TestStreamFailures:
    def test_validation_error_is_plain_json_not_sse(self, client):
        with stream(client, chat_body(temperature=5)) as r:
            assert r.status_code == 400
            assert r.headers["content-type"].startswith("application/json")

    def test_break_mid_stream_sends_error_event_without_done(self, client, admin):
        admin.fault("qwen-mini", mode="stream_break")
        with stream(client, chat_body("one two three four five")) as r:
            assert r.status_code == 200
            events = read_sse(r)
        assert "[DONE]" not in events, "中途断流时不能发 [DONE]，否则客户端会以为回答完整"
        payloads = sse_json(events)
        assert payloads[-1]["error"]["code"] == "stream_interrupted"
        assert any(p.get("choices") for p in payloads[:-1]), "断流前已经收到的内容要正常送达"

    def test_retry_before_first_chunk(self, client, admin):
        admin.fault("qwen-mini", mode="flaky", fail_times=1)
        with stream(client, chat_body("a b")) as r:
            assert r.status_code == 200
            events = read_sse(r)
        assert events[-1] == "[DONE]"
        assert admin.state()["calls"]["qwen-mini"] == 2
