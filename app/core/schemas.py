from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(default="default")
    user_id: str = Field(default="guest")
    message: str


class IntentRequest(BaseModel):
    message: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=10)


class AgentRunRequest(BaseModel):
    session_id: str = Field(default="default")
    user_id: str = Field(default="guest")
    message: str


class IntentResult(BaseModel):
    intent_id: str
    intent_name: str
    category: str
    confidence: float
    slots: dict[str, Any] = Field(default_factory=dict)


class SentimentResult(BaseModel):
    label: str
    score: float


class SearchHit(BaseModel):
    id: str
    title: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None


class ChatResponse(BaseModel):
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
