import json
from typing import Any

from app.services.preprocessor import extract_order_id, extract_product_keyword


INTENT_META: dict[str, dict[str, Any]] = {
    "product_inquiry": {
        "intent_id": "intent_001",
        "intent_name": "商品咨询",
        "category": "business",
        "handler_type": "rag",
        "priority": 5,
    },
    "order_query": {
        "intent_id": "intent_100",
        "intent_name": "订单查询",
        "category": "order",
        "handler_type": "tool",
        "priority": 8,
    },
    "refund_request": {
        "intent_id": "intent_103",
        "intent_name": "申请退款/售后",
        "category": "order",
        "handler_type": "tool",
        "priority": 10,
    },
    "coupon": {
        "intent_id": "intent_005",
        "intent_name": "优惠券使用",
        "category": "business",
        "handler_type": "tool",
        "priority": 4,
    },
    "complaint": {
        "intent_id": "intent_200",
        "intent_name": "投诉",
        "category": "complaint",
        "handler_type": "transfer",
        "priority": 10,
    },
    "human_agent": {
        "intent_id": "intent_203",
        "intent_name": "人工客服",
        "category": "complaint",
        "handler_type": "transfer",
        "priority": 10,
    },
    "greeting": {
        "intent_id": "intent_900",
        "intent_name": "问候",
        "category": "general",
        "handler_type": "llm",
        "priority": 1,
    },
    "fallback": {
        "intent_id": "intent_unknown",
        "intent_name": "未知意图",
        "category": "general",
        "handler_type": "llm",
        "priority": 0,
    },
}


def mock_intent_response(text: str, rule_intent: dict | None = None) -> str:
    lowered = text.lower()
    intent_code = "fallback"
    confidence = 0.55
    reason = "没有命中强规则，交给通用客服回复"

    if any(word in text for word in ["退款", "退货", "赔付", "售后", "坏了", "烂了", "不想要了"]):
        intent_code = "refund_request"
        confidence = 0.92
        reason = "用户表达了退款、退货或售后诉求"
    elif any(word in text for word in ["订单", "物流", "发货", "到哪", "什么时候到", "没收到"]):
        intent_code = "order_query"
        confidence = 0.9
        reason = "用户在询问订单或物流状态"
    elif any(word in text for word in ["商品", "产品", "价格", "库存", "有没有", "能否买", "买", "介绍", "推荐", "苹果", "牛油果", "草莓", "高达", "避孕套"]):
        intent_code = "product_inquiry"
        confidence = 0.88
        reason = "用户在咨询商品信息"
    elif any(word in text for word in ["优惠券", "券", "红包", "满减"]):
        intent_code = "coupon"
        confidence = 0.86
        reason = "用户在询问优惠券或优惠活动"
    elif any(word in text for word in ["投诉", "差评", "太差", "骗人", "生气"]):
        intent_code = "complaint"
        confidence = 0.93
        reason = "用户有明显投诉或强负面表达"
    elif any(word in text for word in ["人工", "客服", "真人", "转人工"]):
        intent_code = "human_agent"
        confidence = 0.9
        reason = "用户明确要求人工客服"
    elif any(word in lowered for word in ["hi", "hello"]) or any(word in text for word in ["你好", "您好", "在吗"]):
        intent_code = "greeting"
        confidence = 0.82
        reason = "用户在问候"

    meta = INTENT_META[intent_code]
    slots: dict[str, Any] = {
        "handler_type": meta["handler_type"],
        "llm_reason": reason,
    }
    order_id = extract_order_id(text)
    product = extract_product_keyword(text) or _extract_requested_product(text)
    if order_id:
        slots["order_id"] = order_id
    if product:
        slots["product"] = product

    return json.dumps(
        {
            "intent_code": intent_code,
            "intent_id": meta["intent_id"],
            "intent_name": meta["intent_name"],
            "category": meta["category"],
            "confidence": confidence,
            "priority": meta["priority"],
            "handler_type": meta["handler_type"],
            "slots": slots,
        },
        ensure_ascii=False,
    )


def _extract_requested_product(text: str) -> str | None:
    for keyword in ["高达", "避孕套", "玩具", "苹果", "牛油果", "草莓"]:
        if keyword in text:
            return keyword
    markers = ["能否买", "有没有", "介绍", "买"]
    for marker in markers:
        if marker in text:
            tail = text.split(marker, 1)[1].strip(" ，。！？?：:")
            return tail[:12] or None
    return None


def mock_chat_response(prompt: str, messages: list[dict[str, str]] | None = None) -> str:
    text = prompt
    if messages:
        text = messages[-1].get("content", "")

    if "替代推荐上下文" in text:
        return "当前没有查到该商品在售。我们主要售卖生鲜食品，可以看看阿克苏苹果、鲜鸡蛋或三文鱼，都是当前可售商品，适合日常补货。"
    if "RAG 商品上下文" in text:
        if "三文鱼" in text:
            return "挪威三文鱼切片 200g 现在有货，肉质细腻、油脂感足，适合刺身、轻煎或搭配沙拉。冷链到家后建议尽快冷藏，开封当天食用口感更好。"
        return "阿克苏苹果主打脆甜多汁，2kg 规格适合家庭日常补货。可以直接鲜食、切水果盘或搭配酸奶；收到后建议冷藏或放阴凉处保存。"
    if "商品推荐上下文" in text:
        if "三文鱼" in text:
            return "挪威三文鱼切片 200g 有货，价格 49.9 元，当前库存 34 件。适合做刺身、轻煎或沙拉，收到后建议冷藏并尽快食用。"
        return "有货的，推荐您优先看库存充足的热销商品；价格和库存都比较稳，收到后建议按页面提示冷藏或尽快食用。"
    if any(word in text for word in ["你好", "您好", "hi", "hello"]):
        return "您好，我是智能客服助手。您可以告诉我订单号、商品名，或直接描述遇到的问题。"
    if any(word in text for word in ["售后", "退款", "退货", "赔付"]):
        return "我理解您想处理售后问题。请提供订单号和商品照片，我会优先帮您发起售后流程。"
    if "知识库内容" in text:
        return "根据知识库信息，我建议您先保留相关凭证，再按页面提示提交售后或咨询客服。"
    return "我已收到您的问题，会结合订单、商品和知识库信息继续为您处理。"


def mock_react_action(context: dict[str, Any]) -> str:
    intent = context["intent"]
    slots = intent.get("slots", {})
    observations = context.get("observations", [])
    done_actions = {item["action"] for item in observations}
    intent_id = intent.get("intent_id")

    if not observations:
        if slots.get("handler_type") == "transfer" or intent_id in {"intent_200", "intent_203"}:
            return _react_json(
                "User needs human support, so create a support ticket.",
                "transfer_human",
                {
                    "user_id": context["user_id"],
                    "session_id": context["session_id"],
                    "message": context["message"],
                    "reason": intent.get("intent_name", "human"),
                },
            )
        if slots.get("handler_type") == "llm":
            return _react_json(
                "This can be answered directly without a business tool.",
                "final",
                final_answer=mock_chat_response("", [{"role": "user", "content": context["message"]}]),
            )
        if intent_id in {"intent_103", "intent_201", "intent_102"}:
            return _react_json(
                "User asks for refund or after-sale support, so call refund_order.",
                "refund_order",
                {"order_id": slots.get("order_id"), "user_id": context["user_id"], "reason": context["message"]},
            )
        if intent_id in {"intent_100", "intent_104", "intent_105", "intent_106"} or slots.get("order_id"):
            return _react_json(
                "User asks about an order, so call query_order.",
                "query_order",
                {"order_id": slots.get("order_id"), "user_id": context["user_id"]},
            )
        if intent_id == "intent_005":
            return _react_json("User asks about coupons, so call list_coupon.", "list_coupon", {"user_id": context["user_id"]})
        if intent_id in {"intent_001", "intent_002", "intent_003", "intent_007"}:
            return _react_json("User asks about product information, so call query_product first.", "query_product", {"product": slots.get("product")})
        return _react_json("No specific tool is obvious, so search the knowledge base.", "search_knowledge", {"query": context["message"], "top_k": 3})

    product_observation = next((item for item in observations if item["action"] == "query_product"), None)
    product_count = len((product_observation or {}).get("result", {}).get("products", []))
    if "query_product" in done_actions and product_count == 0 and "recommend_product" not in done_actions:
        return _react_json("No exact product was found; recommend available alternatives.", "recommend_product", {"product": slots.get("product")})
    if "query_product" in done_actions and product_count > 0 and "search_knowledge" not in done_actions:
        return _react_json("Product data is available; search knowledge base for supporting policy.", "search_knowledge", {"query": context["message"], "top_k": 2})

    return _react_json("The observations are enough to answer.", "final", final_answer="")


def _react_json(thought: str, action: str, action_input: dict[str, Any] | None = None, final_answer: str = "") -> str:
    return json.dumps(
        {
            "thought": thought,
            "action": action,
            "action_input": action_input or {},
            "final_answer": final_answer,
        },
        ensure_ascii=False,
    )
