# 推理网关接口文档

OpenAI 兼容的推理网关。所有错误响应都是 JSON：`{"error": {"message": "...", "type": "...", "param": "...", "code": "..."}}`。

## 鉴权

除 `/healthz` 外，所有接口都要带请求头 `Authorization: Bearer <API Key>`。测试环境有两个 Key：

| Key 代号 | 能访问的模型 |
| --- | --- |
| `valid` | 全部模型 |
| `limited` | 只能访问 `qwen-mini` |

另外用 `invalid` 表示一个不存在的 Key，`missing` 表示不带 Authorization 头。

## 模型

| 模型名 | 类型 | 支持图片输入 |
| --- | --- | --- |
| `qwen-mini` | chat | 否 |
| `qwen-backup` | chat | 否 |
| `qwen-vl` | chat | 是 |
| `embed-mini` | embedding | — |

## 请求处理顺序

网关按下面的顺序检查，**遇到第一个错误就返回**，后面的不再检查：

1. 鉴权：没带 Key 返回 401 `missing_api_key`；Key 不存在返回 401 `invalid_api_key`
2. 请求体不是合法 JSON：400 `invalid_json`
3. 字段校验不通过（缺字段、类型错、取值越界）：400 `invalid_parameter`，`param` 是出错的字段名
4. 模型检查：模型不存在 404 `model_not_found`；Key 无权访问该模型 403 `model_not_allowed`；模型类型和接口不匹配（例如用 embedding 模型调对话接口）400 `model_kind_mismatch`
5. （仅对话接口）`max_tokens` 大于 4096：400 `invalid_parameter`，`param` 为 `max_tokens`
6. （仅对话接口）消息里有图片但模型不支持图片：400 `image_not_supported`
7. （仅对话接口）任意消息内容包含拦截词 `违禁词示例`：400 `content_filter`

## GET /healthz

不需要鉴权。返回 200 `{"status": "ok", "backend": "mock"}`。

## GET /v1/models

返回 200，`data` 是当前 Key 能访问的模型列表。

## POST /v1/chat/completions

| 字段 | 类型 | 必填 | 约束 |
| --- | --- | --- | --- |
| `model` | string | 是 | |
| `messages` | array | 是 | 至少 1 条 |
| `messages[].role` | string | 是 | 只能是 `system`、`user`、`assistant`、`tool` |
| `messages[].content` | string 或 array | 是 | array 时每项是 `{"type": "text", "text": "..."}` 或 `{"type": "image_url", "image_url": {"url": "..."}}` |
| `temperature` | number | 否 | 0 ≤ temperature ≤ 2 |
| `top_p` | number | 否 | 0 ≤ top_p ≤ 1 |
| `max_tokens` | integer | 否 | ≥ 1，且 ≤ 4096 |
| `n` | integer | 否 | 只能是 1 |
| `stream` | boolean | 否 | true 时以 SSE 流式返回，`Content-Type: text/event-stream` |
| `stop` | string 或 string 数组 | 否 | |

成功返回 200：`{"choices": [{"message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}], "usage": {...}}`。

## POST /v1/embeddings

| 字段 | 类型 | 必填 | 约束 |
| --- | --- | --- | --- |
| `model` | string | 是 | 必须是 embedding 模型 |
| `input` | string 或 string 数组 | 是 | 不能为空字符串；数组不能为空、不能含空字符串、最多 256 项 |

成功返回 200：`{"data": [{"embedding": [...], "index": 0}], "usage": {...}}`。
