def _after_product_query(result: dict) -> str:
    """商品查询完成后，根据是否命中商品决定下一状态。"""
    return "product_found" if result.get("products") else "product_missing"


PRODUCT_RAG_ACTIONS = {
    "start": ("query_product",),
    "product_found": ("search_knowledge", "final"),
    "product_missing": ("recommend_product", "final"),
    "completed": ("final",),
}

PRODUCT_RAG_TRANSITIONS = {
    "start": {"query_product": _after_product_query},
    "product_found": {"search_knowledge": "completed"},
    "product_missing": {"recommend_product": "completed"},
}

FLEXIBLE_ROUTE_ACTIONS = {
    # 必须先完成核心业务工具；之后可补充查规则，也可直接组织最终回复。
    "human": {"required": "transfer_human", "next": ("final",)},
    "order": {"required": "query_order", "next": ("search_knowledge", "final")},
    "refund": {"required": "refund_order", "next": ("search_knowledge", "final")},
    "coupon": {"required": "list_coupon", "next": ("search_knowledge", "final")},
    "knowledge": {"required": "search_knowledge", "next": ("final",)},
}


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


def allowed_actions(route: str, observations: list[dict]) -> tuple[str, ...]:
    """根据当前路由阶段，返回唯一合法的下一步 action。"""
    done_actions = {item["action"] for item in observations}

    if route == "product_rag":
        return PRODUCT_RAG_ACTIONS[_product_rag_stage(observations)]

    route_policy = FLEXIBLE_ROUTE_ACTIONS.get(route)
    if route_policy:
        required_action = route_policy["required"]
        if required_action not in done_actions:
            return (required_action,)

        # 可选查询工具只能调用一次；final 始终保留给 LLM 组织自然语言回复。
        return tuple(action for action in route_policy["next"] if action == "final" or action not in done_actions)
    return ("final",)


def _product_rag_stage(observations: list[dict]) -> str:
    """按已完成 action 回放商品状态机，得到当前流程阶段。"""
    stage = "start"
    for observation in observations:
        transition = PRODUCT_RAG_TRANSITIONS.get(stage, {}).get(observation["action"])
        if transition is None:
            continue
        stage = transition(observation["result"]) if callable(transition) else transition
    return stage


def enforce_action(route: str, observations: list[dict], planned_action: str) -> tuple[str, bool]:
    """校验规划动作；越过路由边界时替换为合法 action。"""
    permitted_actions = allowed_actions(route, observations)
    if planned_action in permitted_actions:
        return planned_action, False
    return permitted_actions[0], True
