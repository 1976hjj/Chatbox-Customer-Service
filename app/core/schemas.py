from typing import Any

from pydantic import BaseModel, Field


# 请求/响应数据模型：FastAPI 会用这些类校验入参并生成固定格式的 JSON。
class ChatRequest(BaseModel):
    # /chat 的请求体：一条用户消息加上会话和用户标识。
    session_id: str = Field(default="default")
    user_id: str = Field(default="guest")
    message: str


class IntentRequest(BaseModel):
    # /intent/classify 的请求体：只需要用户原始消息。
    message: str


class SearchRequest(BaseModel):
    # /search 的请求体：top_k 限制在 1 到 10，避免一次返回太多结果。
    query: str
    top_k: int = Field(default=3, ge=1, le=10)


class AgentRunRequest(BaseModel):
    # /agent/run 的请求体：和 ChatRequest 保持一致，便于调试 Agent。
    session_id: str = Field(default="default")
    user_id: str = Field(default="guest")
    message: str


class IntentResult(BaseModel):
    # 意图识别结果：告诉后续流程用户意图、置信度和抽取到的关键信息。
    intent_id: str
    intent_name: str
    category: str
    confidence: float
    slots: dict[str, Any] = Field(default_factory=dict)


class SentimentResult(BaseModel):
    # 情绪分析结果：用于判断是否需要转人工或调整回复策略。
    label: str
    score: float


class SearchHit(BaseModel):
    # 知识库命中结果：包含文档内容、相似度分数和附加元数据。
    id: str
    title: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    # Agent 调用工具的记录：方便排查一次回复具体查了什么数据。
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None


class ChatResponse(BaseModel):
    # 聊天最终响应：把清洗、意图、情绪、工具调用和回复统一返回给前端。
    session_id: str
    user_id: str
    message: str
    normalized_message: str
    sentiment: SentimentResult
    intent: IntentResult
    reply: str
    route: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    citations: list[SearchHit] = Field(default_factory=list)
    react_steps: list[str] = Field(default_factory=list)
    llm: dict[str, Any] = Field(default_factory=dict)
    llm_calls: list[dict[str, Any]] = Field(default_factory=list)
