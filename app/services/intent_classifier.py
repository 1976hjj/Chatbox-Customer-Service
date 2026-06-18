from app.services.preprocessor import extract_order_id, extract_product_keyword
from app.services.llm_service import LLMService


# 意图识别模块：用关键词规则判断用户问题属于商品、订单、售后还是投诉等类别。
INTENTS = [
    {
        "intent_id": "intent_001",
        "intent_name": "商品咨询",
        "category": "business",
        "keywords": ["商品", "价格", "库存", "有没有", "好不好", "新鲜", "推荐", "苹果", "牛油果", "草莓"],
    },
    {
        "intent_id": "intent_002",
        "intent_name": "促销活动",
        "category": "business",
        "keywords": ["活动", "促销", "满减", "折扣", "优惠"],
    },
    {
        "intent_id": "intent_003",
        "intent_name": "配送时效",
        "category": "business",
        "keywords": ["配送", "多久", "什么时候到", "几点送", "时效"],
    },
    {
        "intent_id": "intent_004",
        "intent_name": "支付问题",
        "category": "business",
        "keywords": ["支付", "付款", "扣款", "发票"],
    },
    {
        "intent_id": "intent_005",
        "intent_name": "优惠券使用",
        "category": "business",
        "keywords": ["优惠券", "券", "红包"],
    },
    {
        "intent_id": "intent_006",
        "intent_name": "会员权益",
        "category": "business",
        "keywords": ["会员", "积分", "权益"],
    },
    {
        "intent_id": "intent_007",
        "intent_name": "售后政策",
        "category": "business",
        "keywords": ["售后", "坏果", "坏了", "赔付", "质量"],
    },
    {
        "intent_id": "intent_100",
        "intent_name": "订单查询",
        "category": "order",
        "keywords": ["订单", "查订单", "到哪", "物流", "快递", "没收到"],
    },
    {
        "intent_id": "intent_101",
        "intent_name": "订单修改",
        "category": "order",
        "keywords": ["修改订单", "改订单", "加购"],
    },
    {
        "intent_id": "intent_102",
        "intent_name": "取消订单",
        "category": "order",
        "keywords": ["取消", "不想要"],
    },
    {
        "intent_id": "intent_103",
        "intent_name": "申请退款",
        "category": "order",
        "keywords": ["退款", "退钱", "退货", "赔"],
    },
    {
        "intent_id": "intent_104",
        "intent_name": "物流追踪",
        "category": "order",
        "keywords": ["物流", "快递", "配送员"],
    },
    {
        "intent_id": "intent_105",
        "intent_name": "催促发货",
        "category": "order",
        "keywords": ["催发货", "还不发货", "发货慢"],
    },
    {
        "intent_id": "intent_106",
        "intent_name": "修改地址",
        "category": "order",
        "keywords": ["改地址", "修改地址", "地址错"],
    },
    {
        "intent_id": "intent_200",
        "intent_name": "投诉",
        "category": "complaint",
        "keywords": ["投诉", "差评", "生气", "客服"],
    },
    {
        "intent_id": "intent_201",
        "intent_name": "退款退货",
        "category": "complaint",
        "keywords": ["退货", "退款", "坏了", "烂了"],
    },
    {
        "intent_id": "intent_202",
        "intent_name": "建议反馈",
        "category": "complaint",
        "keywords": ["建议", "反馈", "希望"],
    },
    {
        "intent_id": "intent_203",
        "intent_name": "人工客服",
        "category": "complaint",
        "keywords": ["人工", "转人工", "真人"],
    },
]


def classify_intent(text: str, llm_service: LLMService | None = None) -> dict:
    # 遍历所有意图关键词，选出命中最多的意图作为本次用户问题的分类。
    rule_result = _classify_by_rules(text)
    llm_result = (llm_service or LLMService()).analyze_intent(text, rule_result)
    return _fuse_intents(rule_result, llm_result)


def _classify_by_rules(text: str) -> dict:
    best = None
    best_score = 0
    for intent in INTENTS:
        score = sum(1 for keyword in intent["keywords"] if keyword in text)
        if score > best_score:
            best = intent
            best_score = score

    if best is None:
        best = {
            "intent_id": "intent_unknown",
            "intent_name": "未知意图",
            "category": "general",
            "keywords": [],
        }
        confidence = 0.35
    else:
        confidence = min(0.98, 0.52 + best_score * 0.16)

    # slots 保存从文本里顺手抽出的结构化信息，比如订单号和商品名。
    slots = {}
    order_id = extract_order_id(text)
    product = extract_product_keyword(text)
    if order_id:
        slots["order_id"] = order_id
    if product:
        slots["product"] = product

    return {
        "intent_id": best["intent_id"],
        "intent_name": best["intent_name"],
        "category": best["category"],
        "confidence": round(confidence, 3),
        "slots": slots,
    }


def _fuse_intents(rule_result: dict, llm_result: dict) -> dict:
    rule_confidence = float(rule_result.get("confidence", 0))
    llm_confidence = float(llm_result.get("confidence", 0))
    rule_priority = _intent_priority(rule_result.get("intent_id"))
    llm_priority = _parse_priority(llm_result.get("priority"), llm_result.get("intent_code"))
    rule_unknown = rule_result.get("intent_id") == "intent_unknown"

    if rule_unknown or llm_confidence >= rule_confidence + 0.12 or (
        llm_priority >= rule_priority + 3 and llm_confidence >= 0.85
    ):
        llm_meta = _llm_code_meta(llm_result.get("intent_code"))
        selected = {
            "intent_id": llm_meta.get("intent_id") or llm_result.get("intent_id", "intent_unknown"),
            "intent_name": llm_meta.get("intent_name") or llm_result.get("intent_name", "未知意图"),
            "category": llm_meta.get("category") or llm_result.get("category", "general"),
            "confidence": round(llm_confidence, 3),
            "slots": dict(llm_result.get("slots") or {}),
        }
        selected["slots"]["handler_type"] = llm_meta.get("handler_type") or selected["slots"].get("handler_type")
        selected["slots"]["intent_source"] = "llm"
        selected["slots"]["rule_intent_id"] = rule_result.get("intent_id")
        selected["slots"]["llm_intent_code"] = llm_result.get("intent_code")
        return selected

    merged_slots = dict(rule_result.get("slots") or {})
    merged_slots.setdefault("handler_type", llm_result.get("handler_type"))
    merged_slots["intent_source"] = "rule"
    merged_slots["llm_intent_code"] = llm_result.get("intent_code")
    return {
        **rule_result,
        "slots": merged_slots,
    }


def _intent_priority(intent_id: str | None) -> int:
    if intent_id in {"intent_103", "intent_201", "intent_102", "intent_200", "intent_203"}:
        return 10
    if intent_id in {"intent_100", "intent_104", "intent_105", "intent_106"}:
        return 8
    if intent_id in {"intent_001", "intent_002", "intent_003", "intent_007"}:
        return 5
    if intent_id == "intent_005":
        return 4
    return 0


def _llm_code_priority(intent_code: str | None) -> int:
    if intent_code in {"refund_request", "complaint", "human_agent"}:
        return 10
    if intent_code == "order_query":
        return 8
    if intent_code == "product_inquiry":
        return 5
    if intent_code == "coupon":
        return 4
    return 0


def _parse_priority(priority: object, intent_code: str | None) -> int:
    if isinstance(priority, int):
        return priority
    if isinstance(priority, str):
        if priority.isdigit():
            return int(priority)
        label_scores = {"high": 10, "medium": 5, "low": 1}
        if priority.lower() in label_scores:
            return label_scores[priority.lower()]
    return _llm_code_priority(intent_code)


def _llm_code_meta(intent_code: str | None) -> dict:
    mapping = {
        "product_inquiry": {"intent_id": "intent_001", "intent_name": "商品咨询", "category": "business", "handler_type": "rag"},
        "order_query": {"intent_id": "intent_100", "intent_name": "订单查询", "category": "order", "handler_type": "tool"},
        "refund_request": {"intent_id": "intent_103", "intent_name": "申请退款/售后", "category": "order", "handler_type": "tool"},
        "coupon": {"intent_id": "intent_005", "intent_name": "优惠券使用", "category": "business", "handler_type": "tool"},
        "complaint": {"intent_id": "intent_200", "intent_name": "投诉", "category": "complaint", "handler_type": "transfer"},
        "human_agent": {"intent_id": "intent_203", "intent_name": "人工客服", "category": "complaint", "handler_type": "transfer"},
        "greeting": {"intent_id": "intent_900", "intent_name": "问候", "category": "general", "handler_type": "llm"},
        "fallback": {"intent_id": "intent_unknown", "intent_name": "未知意图", "category": "general", "handler_type": "llm"},
    }
    return mapping.get(intent_code or "", {})
