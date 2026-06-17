import re


SENSITIVE_WORDS = {
    "傻逼": "***",
    "垃圾客服": "客服体验不好",
}


def normalize_message(message: str) -> str:
    text = message.strip()
    text = re.sub(r"\s+", " ", text)
    for word, replacement in SENSITIVE_WORDS.items():
        text = text.replace(word, replacement)
    return text


def extract_order_id(text: str) -> str | None:
    match = re.search(r"\bO\d{8,}\b", text, flags=re.IGNORECASE)
    return match.group(0).upper() if match else None


def extract_product_keyword(text: str) -> str | None:
    products = ["苹果", "牛油果", "鸡蛋", "草莓", "三文鱼", "西兰花", "香蕉", "番茄"]
    for product in products:
        if product in text:
            return product
    return None
