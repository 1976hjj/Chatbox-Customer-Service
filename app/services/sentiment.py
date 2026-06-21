NEGATIVE_WORDS = {"坏了", "退款", "投诉", "生气", "差评", "没收到", "烂", "赔", "取消", "太慢"}
POSITIVE_WORDS = {"谢谢", "不错", "满意", "好吃", "新鲜", "推荐", "喜欢"}


def analyze_sentiment(text: str) -> dict:
    # 情绪分析模块：比较正向词和负向词命中数，输出情绪标签和一个简单置信分。
    negative_hits = sum(1 for word in NEGATIVE_WORDS if word in text)
    positive_hits = sum(1 for word in POSITIVE_WORDS if word in text)

    if negative_hits > positive_hits:
        label = "negative"
        score = min(0.95, 0.55 + negative_hits * 0.12)
    elif positive_hits > negative_hits:
        label = "positive"
        score = min(0.95, 0.55 + positive_hits * 0.12)
    else:
        label = "neutral"
        score = 0.5

    return {"label": label, "score": round(score, 3)}
