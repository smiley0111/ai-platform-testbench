"""请求体校验，字段和取值范围对齐 OpenAI Chat Completions / Embeddings。"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TextPart(BaseModel):
    type: Literal["text"]
    text: str


class ImageURL(BaseModel):
    url: str


class ImagePart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURL


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart]

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return " ".join(p.text for p in self.content if isinstance(p, TextPart))

    def has_image(self) -> bool:
        return isinstance(self.content, list) and any(isinstance(p, ImagePart) for p in self.content)


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[Message] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    n: int = Field(default=1, ge=1, le=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    input: str | list[str]

    @field_validator("input")
    @classmethod
    def _not_empty(cls, v: str | list[str]) -> str | list[str]:
        items = [v] if isinstance(v, str) else v
        if not items or any(not s for s in items):
            raise ValueError("input must be a non-empty string or a list of non-empty strings")
        if len(items) > 256:
            raise ValueError("input list must not exceed 256 items")
        return v

    def texts(self) -> list[str]:
        return [self.input] if isinstance(self.input, str) else list(self.input)
