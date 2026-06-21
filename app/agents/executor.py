from typing import Any

from app.services.tools import TOOL_REGISTRY


def build_action_input(
    action: str,
    action_input: dict[str, Any],
    session_id: str,
    user_id: str,
    message: str,
    intent: dict,
) -> dict[str, Any]:
    """在调用已注册工具前，补齐并规范规划器给出的动作参数。"""
    slots = intent.get("slots", {})

    # 优先使用意图分类阶段提取出的结构化 slots；字段缺失时，
    # 再回退使用规划器传入的 action_input。
    if action == "query_product":
        return {"product": slots.get("product") or action_input.get("product")}
    if action == "search_knowledge":
        return {"query": action_input.get("query") or message, "top_k": action_input.get("top_k", 3)}
    if action == "query_order":
        return {"order_id": slots.get("order_id") or action_input.get("order_id"), "user_id": user_id}
    if action == "refund_order":
        return {
            "order_id": slots.get("order_id") or action_input.get("order_id"),
            "user_id": user_id,
            "reason": action_input.get("reason") or message,
        }
    if action == "list_coupon":
        return {"user_id": user_id}
    if action == "transfer_human":
        return {
            "user_id": user_id,
            "session_id": session_id,
            "message": action_input.get("message") or message,
            "reason": action_input.get("reason") or intent.get("intent_name", "human"),
        }
    return action_input


def call_tool(name: str, **kwargs) -> dict:
    """通过工具注册表执行一个工具，并保留原有的调用数据结构。"""
    result = TOOL_REGISTRY[name](**kwargs)
    return {"name": name, "arguments": kwargs, "result": result}


def summarize_observation(action: str, result: dict) -> str:
    """将工具结果转换为 react_steps 中使用的简短 Observation 文本。"""
    # Observation 有意保持简短：让下一步规划知道发生了什么，
    # 同时避免把完整工具结果塞进 react_steps。
    if action == "query_product":
        return f"products={len(result.get('products', []))}"
    if action == "recommend_product":
        return f"recommendations={len(result.get('recommendations', []))}"
    if action == "search_knowledge":
        return f"hits={len(result.get('hits', []))}"
    if action == "query_order":
        return "order=found" if result.get("order") else "order=missing"
    if action == "refund_order":
        return f"accepted={result.get('accepted')}"
    if action == "list_coupon":
        return f"coupons={len(result.get('coupons', []))}"
    if action == "transfer_human":
        return f"ticket={result.get('ticket', {}).get('ticket_id')}"
    return "ok"
