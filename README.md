# ai-platform-testbench

AI 平台测试工作台。按 AI 开发平台的四大场景组织测试：模型 API、模型推理、模型训练、Notebook。

被测对象是一个自己写的 **OpenAI 兼容推理网关**，后端可以接本地 Ollama（真实模型），也可以接一个
**能注入故障的 Mock 后端**。可靠性测试（重试、超时、熔断、降级）必须能稳定复现故障，真实模型做不到，
所以默认用 Mock 跑，真实模型的用例单独标记。

## 进度

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 1 | 模型 API 测试：功能、流式、协议兼容、安全、可靠性 | 已完成 |
| 2 | 性能基准：首字延迟、吞吐、并发上限（Locust） | 未开始 |
| 3 | 训练场景：数据并行一致性、Checkpoint 保存与恢复 | 未开始 |
| 4 | Notebook 场景：Jupyter 内核管理 | 未开始 |
| 5 | AI 辅助测试：大模型根据接口文档生成用例 | 未开始 |
| 6 | Kubernetes 部署：弹性伸缩、灰度发布（GitHub Actions + kind） | 未开始 |

## 运行

```bash
uv sync --extra dev
uv run pytest                 # 全部用例，Mock 后端
uv run pytest -m "not ollama" # 只跑不依赖真实模型的用例
```

启动网关（默认 Mock 后端，开发用 key 为 `sk-local-dev`）：

```bash
uv run python -m testbench.gateway.app
```

接 Ollama：

```bash
ollama pull qwen2.5:0.5b
TESTBENCH_BACKEND=ollama uv run python -m testbench.gateway.app
```

## 网关做了什么

请求依次经过：

1. **鉴权**：`Authorization: Bearer <key>`，缺失或错误返回 401
2. **限流**：每个 key 一个令牌桶，超出返回 429 和 `Retry-After`
3. **参数校验**：字段和取值范围对齐 OpenAI，错误返回 400，`param` 指出是哪个字段
4. **模型权限（RBAC）**：key 只能访问被授权的模型，否则 403
5. **内容过滤**：输入命中拦截词返回 400；输出命中则打码，`finish_reason` 置为 `content_filter`
6. **重试 / 超时**：后端 5xx 和 429 指数退避重试，4xx 不重试；超时返回 504
7. **熔断**：连续失败达到阈值后打开，快速返回 503；一段时间后半开，放一个试探请求
8. **降级**：主模型失败或熔断时切到备用模型，响应头 `x-fallback: true`、`x-served-model` 标明实际模型

所有错误都是 OpenAI 的格式 `{"error": {"message", "type", "param", "code"}}`，官方 SDK 能识别成对应的异常类型。

## 测试设计

### 阶段 1：模型 API

| 文件 | 测什么 | 关键用例 |
| --- | --- | --- |
| `test_functional.py` | 功能正确性 | 响应结构；usage 求和；多轮对话；`max_tokens` 截断与 `finish_reason=length`；`stop`；11 种非法参数；边界值；多模态输入；Embeddings 顺序、维度、确定性 |
| `test_streaming.py` | 流式 SSE | 事件格式与 `[DONE]`；所有 chunk 同一个 id；流式拼接结果等于非流式；`include_usage`；**中途断流不能发 `[DONE]`**；首包前失败可重试 |
| `test_compat.py` | 协议兼容 | 官方 `openai` SDK 只改 `base_url` 就能调用对话、流式、Embeddings、模型列表；401/403/404/400/429 被 SDK 识别成正确的异常类 |
| `test_security.py` | 安全性 | 5 种鉴权失败；**鉴权先于参数校验**；错误信息不回显 key；RBAC；限流按 key 隔离；**非法请求也消耗配额**；输入、多模态、输出、流式输出的内容过滤 |
| `test_reliability.py` | 可靠性 | 瞬时错误重试成功；超过次数放弃；4xx 不重试；超时按时返回；熔断打开后快速失败且不再打后端；半开成功恢复、失败重新打开；熔断按模型隔离；降级与流式降级；50 个并发请求 |

几个值得注意的设计点（也是用例里的断言）：

- **中途断流时不能发 `[DONE]`。** 否则客户端会把半截回答当成完整回答展示给用户。
- **鉴权和限流在参数校验之前。** 否则未授权的人能探测接口行为，攻击者也能用错误请求无限刷接口。
- **降级要如实告诉用户。** 响应里的 `model` 字段和响应头都要标明实际是哪个模型回答的。
- **测试里关掉 SDK 的自动重试。** `openai` SDK 默认对 429 和 5xx 自动重试 2 次，不关掉就测不到网关的真实行为。

### 已知限制

- 流式输出的内容过滤按 chunk 检查，拦截词如果被切在两个 chunk 之间会漏掉。真实系统需要维护一个滑动窗口缓冲。
- Mock 后端按空格切分 token，只用于验证网关逻辑，不代表真实模型的 token 计数。
