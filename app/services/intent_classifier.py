from app.services.preprocessor import extract_order_id, extract_product_keyword


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


def classify_intent(text: str) -> dict:
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
