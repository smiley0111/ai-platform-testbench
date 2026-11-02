# AI 辅助测试评测：大模型根据接口文档生成用例

被测对象是阶段 1 的推理网关（Mock 后端）。模型只拿到 `src/testbench/aitest/api_spec.md` 这份接口文档，按 5 个测试类别分别生成用例（每类要求 8 条），输出为 JSON 数据，由执行器统一发请求和比对。

## 指标说明

- **格式不合法**：缺字段、路径不在文档里、POST 没有请求体等，执行前就被丢弃
- **有效**：在正确实现的网关上执行通过，即期望结果写对了。期望写错的用例在回归里只会制造误报
- **需求覆盖**：接口文档拆成 26 条需求点，被至少一条有效用例命中才算覆盖（按实际行为判断，不看用例名字）
- **变异得分**：往网关里植入 14 个真实可能出现的 bug（每次只改一处），有效用例能发现几个

## 总览

| 来源 | 生成耗时 | 输出 token | 生成条数 | 格式不合法 | 重复 | 执行条数 | 有效 | 有效率 | 需求覆盖 | 变异得分 |
|---|---|---|---|---|---|---|---|---|---|---|
| 人工编写（基准） | 0s | 0 | 30 | 0 | 0 | 30 | 30 | 100% | 26/26（100%） | 14/14（100%） |
| qwen2.5:0.5b 约束解码 | 378s | 6646 | 33 | 3 | 13 | 17 | 6 | 35% | 1/26（4%） | 1/14（7%） |
| qwen2.5:0.5b 无约束 | 390s | 8681 | 24 | 6 | 7 | 11 | 4 | 36% | 2/26（8%） | 1/14（7%） |
| qwen2.5:1.5b 约束解码 | 593s | 6836 | 36 | 10 | 8 | 18 | 15 | 83% | 11/26（42%） | 7/14（50%） |
| qwen2.5:1.5b 无约束 | 549s | 6732 | 33 | 1 | 9 | 23 | 13 | 57% | 9/26（35%） | 5/14（36%） |
| qwen2.5:3b 约束解码 | 530s | 3544 | 38 | 22 | 3 | 13 | 8 | 62% | 9/26（35%） | 3/14（21%） |

## 需求覆盖明细

| 需求 | 人工编写（基准） | qwen2.5:0.5b 约束解码 | qwen2.5:0.5b 无约束 | qwen2.5:1.5b 约束解码 | qwen2.5:1.5b 无约束 | qwen2.5:3b 约束解码 |
|---|---|---|---|---|---|---|
| R01 healthz 无需鉴权返回 200 | ✓ |  |  |  | ✓ |  |
| R02 模型列表返回 200 | ✓ |  |  |  |  |  |
| R03 模型列表缺 Key 返回 401 | ✓ |  |  |  |  |  |
| R04 对话非流式成功 | ✓ |  | ✓ | ✓ | ✓ | ✓ |
| R05 对话流式成功（SSE） | ✓ |  |  | ✓ | ✓ |  |
| R06 对话缺 Key 401 | ✓ |  |  |  |  |  |
| R07 对话 Key 无效 401 | ✓ |  |  | ✓ |  |  |
| R08 请求体不是合法 JSON 400 | ✓ |  |  | ✓ |  | ✓ |
| R09 messages 缺失或为空 400 | ✓ |  |  |  |  | ✓ |
| R10 model 缺失 400 | ✓ |  |  |  |  |  |
| R11 temperature 越界 400 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| R12 top_p 越界 400 | ✓ |  |  |  |  | ✓ |
| R13 max_tokens 越界 400 | ✓ |  |  | ✓ | ✓ | ✓ |
| R14 n 不为 1 时 400 | ✓ |  |  |  |  | ✓ |
| R15 非法 role 400 | ✓ |  |  |  |  | ✓ |
| R16 模型不存在 404 | ✓ |  |  | ✓ | ✓ |  |
| R17 Key 无权访问模型 403 | ✓ |  |  |  |  |  |
| R18 对话接口用 embedding 模型 400 | ✓ |  |  |  |  |  |
| R19 不支持图片的模型收到图片 400 | ✓ |  |  |  |  |  |
| R20 输入命中拦截词 400 | ✓ |  |  | ✓ | ✓ |  |
| R21 取值边界内的参数被接受（如 temperature=2、max_tokens=4096） | ✓ |  |  |  |  | ✓ |
| R22 视觉模型接受图片输入 | ✓ |  |  |  |  |  |
| R23 Embeddings 成功 | ✓ |  |  | ✓ | ✓ |  |
| R24 Embeddings input 为空 400 | ✓ |  |  | ✓ |  |  |
| R25 Embeddings 用对话模型 400 | ✓ |  |  | ✓ | ✓ |  |
| R26 Embeddings 缺 Key 401 | ✓ |  |  |  |  |  |

## 变异体明细（✓ = 被发现）

| 植入的 bug | 人工编写（基准） | qwen2.5:0.5b 约束解码 | qwen2.5:0.5b 无约束 | qwen2.5:1.5b 约束解码 | qwen2.5:1.5b 无约束 | qwen2.5:3b 约束解码 |
|---|---|---|---|---|---|---|
| M01 temperature 上界写成 3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| M02 top_p 上界写成 1.5 | ✓ |  |  |  |  |  |
| M03 n 允许大于 1 | ✓ |  |  |  |  | ✓ |
| M04 messages 允许为空 | ✓ |  |  |  |  |  |
| M05 max_tokens 上限差一（4096 被拒绝） | ✓ |  |  |  |  |  |
| M06 RBAC 失效：受限 Key 能访问所有模型 | ✓ |  |  |  |  |  |
| M07 模型类型不检查 | ✓ |  |  | ✓ | ✓ |  |
| M08 内容过滤失效 | ✓ |  |  | ✓ | ✓ |  |
| M09 图片输入不检查 | ✓ |  |  |  |  |  |
| M10 Key 无效时返回 403 而不是 401 | ✓ |  |  | ✓ | ✓ |  |
| M11 非法 JSON 未处理（变成 500） | ✓ |  |  | ✓ |  | ✓ |
| M12 Embeddings 不拒绝空输入 | ✓ |  |  | ✓ |  |  |
| M13 模型列表不鉴权 | ✓ |  |  |  |  |  |
| M14 流式请求按非流式返回 | ✓ |  |  | ✓ | ✓ |  |

## 期望写错的用例：常见模式

**qwen2.5:0.5b 约束解码**（共 11 条）

- 期望 400 content_filter → 实际 400 invalid_parameter：4 条
- 期望 400 model_kind_mismatch → 实际 400 invalid_parameter：2 条
- 期望 400 invalid_parameter → 实际 200 ：1 条
- 期望 400 model_not_allowed → 实际 200 ：1 条
- 期望 400 image_not_supported → 实际 400 invalid_parameter：1 条
- 期望 404 model_not_found → 实际 400 invalid_parameter：1 条

**qwen2.5:0.5b 无约束**（共 7 条）

- 期望 400 content_filter → 实际 400 invalid_parameter：2 条
- 期望 400 invalid_parameter → 实际 200 ：1 条
- 期望 400 model_not_allowed → 实际 200 ：1 条
- 期望 404 model_not_found → 实际 400 invalid_parameter：1 条
- 期望 400 model_kind_mismatch → 实际 400 invalid_parameter：1 条
- 期望 200 → 实际 405 unparseable：1 条

**qwen2.5:1.5b 约束解码**（共 3 条）

- 期望 400 model_kind_mismatch → 实际 200 ：1 条
- 期望 401 invalid_api_key → 实际 401 missing_api_key：1 条
- 期望 401 invalid_api_key → 实际 403 model_not_allowed：1 条

**qwen2.5:1.5b 无约束**（共 10 条）

- 期望 400 model_kind_mismatch → 实际 200 ：2 条
- 期望 400 invalid_parameter → 实际 200 ：2 条
- 期望 401 missing_api_key → 实际 200 ：1 条
- 期望 401 invalid_api_key → 实际 200 ：1 条
- 期望 403 model_not_allowed → 实际 200 ：1 条
- 期望 404 model_not_found → 实际 200 ：1 条

**qwen2.5:3b 约束解码**（共 5 条）

- 期望 400 image_not_supported → 实际 200 ：2 条
- 期望 400 model_kind_mismatch → 实际 200 ：1 条
- 期望 400 content_filter → 实际 200 ：1 条
- 期望 200 → 实际 400 model_kind_mismatch：1 条

## 格式不合法的原因

**qwen2.5:0.5b 约束解码**：GET request with a body（2）；output is not JSON: Unterminated string starting at（1）

**qwen2.5:0.5b 无约束**：GET request with a body（4）；output is not JSON: Unterminated string starting at（2）

**qwen2.5:1.5b 约束解码**：POST request without body (use raw_body "" to test an empty body)（9）；output is not JSON: Unterminated string starting at（1）

**qwen2.5:1.5b 无约束**：output is not JSON: Unterminated string starting at（1）

**qwen2.5:3b 约束解码**：POST request without body (use raw_body "" to test an empty body)（22）

## 原始数据

`reports/aitest/raw/` 是模型的原始输出，`reports/aitest/cases-*.json` 是解析去重后的用例。
