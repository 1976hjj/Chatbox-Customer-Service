import re


# 消息预处理模块：负责把用户输入清洗成更稳定、适合后续匹配的文本。
SENSITIVE_WORDS = {
    "傻逼": "***",
    "垃圾客服": "客服体验不好",
}


def normalize_message(message: str) -> str:
    # 去掉首尾空白、合并多余空格，并把敏感表达替换成更温和的说法。
    text = message.strip()
    text = re.sub(r"\s+", " ", text)
    for word, replacement in SENSITIVE_WORDS.items():
        text = text.replace(word, replacement)
    return text


def extract_order_id(text: str) -> str | None:
    # 从文本里提取 O 开头的订单号，找不到就返回 None。
    match = re.search(r"\bO\d{8,}\b", text, flags=re.IGNORECASE)
    return match.group(0).upper() if match else None


def extract_product_keyword(text: str) -> str | None:
    # 从商品关键词表里找用户提到的第一个商品名。
    products = ["苹果", "牛油果", "鸡蛋", "草莓", "三文鱼", "西兰花", "香蕉", "番茄"]
    for product in products:
        if product in text:
            return product
    return None
