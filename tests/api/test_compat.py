"""协议兼容性：用户用官方 openai SDK，只改 base_url 就能接入，错误也能被 SDK 正确识别。"""

import httpx
import openai
import pytest
from conftest import KEY_ALL, KEY_LIMITED, KEY_RATE


def sdk(base_url: str, key: str = KEY_ALL) -> openai.OpenAI:
    # SDK 默认会对 429 / 5xx 自动重试 2 次，测试里关掉，才能看到网关的真实行为
    return openai.OpenAI(base_url=f"{base_url}/v1", api_key=key, max_retries=0,
                         http_client=httpx.Client(trust_env=False))


def test_sdk_chat(base_url):
    resp = sdk(base_url).chat.completions.create(
        model="qwen-mini", messages=[{"role": "user", "content": "hello sdk"}])
    assert resp.choices[0].message.content
    assert resp.usage.total_tokens > 0


def test_sdk_stream(base_url):
    chunks = list(sdk(base_url).chat.completions.create(
        model="qwen-mini", messages=[{"role": "user", "content": "hello stream"}], stream=True))
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert "hello stream" in text
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_sdk_stream_with_usage(base_url):
    chunks = list(sdk(base_url).chat.completions.create(
        model="qwen-mini", messages=[{"role": "user", "content": "x"}], stream=True,
        stream_options={"include_usage": True}))
    assert chunks[-1].usage is not None


def test_sdk_embeddings(base_url):
    resp = sdk(base_url).embeddings.create(model="embed-mini", input=["a", "b"])
    assert len(resp.data) == 2 and len(resp.data[0].embedding) > 0


def test_sdk_models_list(base_url):
    assert "qwen-mini" in {m.id for m in sdk(base_url).models.list()}


def test_sdk_extra_body_is_ignored(base_url):
    resp = sdk(base_url).chat.completions.create(
        model="qwen-mini", messages=[{"role": "user", "content": "x"}],
        extra_body={"vendor_specific_option": True})
    assert resp.choices[0].message.content


@pytest.mark.parametrize("key, kwargs, error", [
    ("sk-wrong", {}, openai.AuthenticationError),
    (KEY_ALL, {"model": "gpt-99"}, openai.NotFoundError),
    (KEY_ALL, {"temperature": 3}, openai.BadRequestError),
    (KEY_LIMITED, {"model": "qwen-vl"}, openai.PermissionDeniedError),
], ids=["401-auth", "404-model", "400-param", "403-permission"])
def test_sdk_error_mapping(base_url, key, kwargs, error):
    params = {"model": "qwen-mini", "messages": [{"role": "user", "content": "x"}], **kwargs}
    with pytest.raises(error) as exc:
        sdk(base_url, key).chat.completions.create(**params)
    assert exc.value.code is not None


def test_sdk_rate_limit_error(base_url):
    client = sdk(base_url, KEY_RATE)
    params = {"model": "qwen-mini", "messages": [{"role": "user", "content": "x"}]}
    for _ in range(3):
        client.chat.completions.create(**params)
    with pytest.raises(openai.RateLimitError) as exc:
        client.chat.completions.create(**params)
    assert exc.value.response.headers["retry-after"]
