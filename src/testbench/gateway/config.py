"""网关配置。对外的模型名和后端真实模型名分开，便于同一套测试换后端跑。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ModelSpec:
    backend_name: str
    kind: str = "chat"  # chat | embedding
    vision: bool = False
    fallback: str | None = None


@dataclass
class KeyPolicy:
    models: set[str] | None = None  # None 表示可访问全部模型
    rpm: int = 600


@dataclass
class GatewayConfig:
    backend: str = "mock"  # mock | ollama
    ollama_url: str = "http://localhost:11434"
    keys: dict[str, KeyPolicy] = field(default_factory=dict)
    models: dict[str, ModelSpec] = field(default_factory=dict)
    request_timeout: float = 60.0
    max_retries: int = 2
    retry_backoff: float = 0.2
    breaker_failures: int = 5
    breaker_reset: float = 10.0
    max_tokens_limit: int = 4096
    blocked_words: list[str] = field(default_factory=list)
    admin_token: str | None = None  # 设置后才开放 /admin 故障注入接口
    version: str = "dev"  # 写进响应头，灰度发布时用来区分新旧版本
    startup_delay: float = 0.0  # 启动后多久才就绪，模拟加载模型
    fail_ready: bool = False  # 模拟一个就绪检查永远失败的坏版本
    mock_latency: float = 0.0  # Mock 后端每次调用的耗时

    @classmethod
    def default(cls) -> GatewayConfig:
        chat_model = os.getenv("TESTBENCH_CHAT_MODEL", "qwen2.5:0.5b")
        embed_model = os.getenv("TESTBENCH_EMBED_MODEL", "nomic-embed-text")
        keys = {os.getenv("TESTBENCH_API_KEY", "sk-local-dev"): KeyPolicy(
            rpm=int(os.getenv("TESTBENCH_RPM", "600")))}
        for item in filter(None, os.getenv("TESTBENCH_EXTRA_KEYS", "").split(",")):
            key, rpm = item.rsplit(":", 1)  # 格式：key:rpm,key:rpm
            keys[key] = KeyPolicy(rpm=int(rpm))
        return cls(
            backend=os.getenv("TESTBENCH_BACKEND", "mock"),
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            keys=keys,
            request_timeout=float(os.getenv("TESTBENCH_TIMEOUT", "60")),
            models={
                "qwen-mini": ModelSpec(chat_model, fallback="qwen-backup"),
                "qwen-backup": ModelSpec(chat_model),
                "embed-mini": ModelSpec(embed_model, kind="embedding"),
            },
            blocked_words=["违禁词示例"],
            admin_token=os.getenv("TESTBENCH_ADMIN_TOKEN"),
            version=os.getenv("TESTBENCH_VERSION", "dev"),
            startup_delay=float(os.getenv("TESTBENCH_STARTUP_DELAY", "0")),
            fail_ready=os.getenv("TESTBENCH_FAIL_READY", "") == "1",
            mock_latency=float(os.getenv("TESTBENCH_MOCK_LATENCY", "0")),
        )
