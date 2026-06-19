def route_intent(intent: dict, sentiment: dict) -> str:
    """根据业务决策字段确定处理路由。

    这个函数只负责路由决策：不调用工具、不生成回复，也不修改意图或情绪数据。
    """
    handler_type = intent["slots"].get("handler_type")

    # LLM 给出的意图元数据优先级最高，因为它明确指定本轮应由哪类处理器接管。
    if handler_type == "transfer":
        return "human"
    if handler_type == "llm":
        return "llm"

    # 即使文本分类器无法确定意图，提取到订单号也是强信号，优先进入订单工具，
    # 而不是走通用知识问答。
    if intent["slots"].get("order_id") and intent["intent_id"] == "intent_unknown":
        return "order"

    # 投诉且情绪为负面时，转交人工客服处理。
    if sentiment["label"] == "negative" and intent["category"] == "complaint":
        return "human"

    # 其余分支将稳定的意图 ID 映射到对应业务路由。
    if intent["intent_id"] in {"intent_001", "intent_002", "intent_003", "intent_007"}:
        return "product_rag"
    if intent["intent_id"] in {"intent_100", "intent_104", "intent_105", "intent_106"}:
        return "order"
    if intent["intent_id"] in {"intent_103", "intent_201", "intent_102"}:
        return "refund"
    if intent["intent_id"] == "intent_005":
        return "coupon"
    if intent["intent_id"] in {"intent_200", "intent_203"}:
        return "human"
    return "knowledge"
