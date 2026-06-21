from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """CustomerServiceAgent 中一次对话流转的数据结构说明。

    为兼容现有调用，当前 API 仍返回字典。这个类以轻量方式标明业务决策、
    执行、调试和最终响应四层字段。
    """

    session_id: str
    user_id: str
    message: str

    # 业务决策字段。
    normalized_message: str = ""
    intent: dict[str, Any] = field(default_factory=dict)
    sentiment: dict[str, Any] = field(default_factory=dict)
    route: str = ""

    # 执行字段。
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)

    # 调试字段。
    react_steps: list[str] = field(default_factory=list)
    llm_trace: list[dict[str, Any]] = field(default_factory=list)

    # 最终输出字段。
    reply: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    handler_type: str = ""
