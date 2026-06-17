from datetime import datetime

from app.repositories import mock_db
from app.services.intent_classifier import classify_intent
from app.services.preprocessor import normalize_message
from app.services.sentiment import analyze_sentiment
from app.services.tools import TOOL_REGISTRY


class CustomerServiceAgent:
    def run(self, session_id: str, user_id: str, message: str) -> dict:
        normalized = normalize_message(message)
        sentiment = analyze_sentiment(normalized)
        intent = classify_intent(normalized)
        tool_calls = []
        citations = []

        route = self._route(intent, sentiment)
        reply = ""

        if route == "product_rag":
            product = intent["slots"].get("product")
            tool_calls.append(self._call("query_product", product=product))
            tool_calls.append(self._call("search_knowledge", query=normalized, top_k=2))
            citations = tool_calls[-1]["result"]["hits"]
            reply = self._product_reply(product, tool_calls[0]["result"], citations)
        elif route == "order":
            order_id = intent["slots"].get("order_id")
            tool_calls.append(self._call("query_order", order_id=order_id, user_id=user_id))
            reply = self._order_reply(tool_calls[0]["result"])
        elif route == "refund":
            order_id = intent["slots"].get("order_id")
            tool_calls.append(self._call("refund_order", order_id=order_id, user_id=user_id, reason=normalized))
            reply = self._refund_reply(tool_calls[0]["result"])
        elif route == "coupon":
            tool_calls.append(self._call("list_coupon", user_id=user_id))
            reply = self._coupon_reply(tool_calls[0]["result"])
        elif route == "human":
            tool_calls.append(
                self._call(
                    "transfer_human",
                    user_id=user_id,
                    session_id=session_id,
                    message=normalized,
                    reason=intent["intent_name"],
                )
            )
            reply = "已为您创建人工客服工单，客服会优先处理。您也可以继续补充订单号、商品照片或问题细节。"
        else:
            tool_calls.append(self._call("search_knowledge", query=normalized, top_k=3))
            citations = tool_calls[-1]["result"]["hits"]
            reply = self._knowledge_reply(citations)

        record = {
            "session_id": session_id,
            "user_id": user_id,
            "message": message,
            "normalized_message": normalized,
            "intent": intent,
            "sentiment": sentiment,
            "reply": reply,
            "route": route,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        mock_db.save_chat(record)
        return {**record, "tool_calls": tool_calls, "citations": citations}

    def _route(self, intent: dict, sentiment: dict) -> str:
        if intent["slots"].get("order_id") and intent["intent_id"] == "intent_unknown":
            return "order"
        if sentiment["label"] == "negative" and intent["category"] == "complaint":
            return "human"
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

    def _call(self, name: str, **kwargs) -> dict:
        result = TOOL_REGISTRY[name](**kwargs)
        return {"name": name, "arguments": kwargs, "result": result}

    def _product_reply(self, product: str | None, product_result: dict, hits: list[dict]) -> str:
        products = product_result["products"]
        if not products:
            return f"暂时没有查到和{product or '该商品'}相关的在售商品。您可以换个关键词，我也可以为您推荐相近商品。"
        lines = ["为您查到以下商品："]
        for item in products[:3]:
            stock_text = "有货" if item["stock"] > 0 else "暂时缺货"
            lines.append(f"- {item['name']}，价格{item['price']}元，库存{item['stock']}，{stock_text}")
        if hits:
            lines.append(f"补充说明：{hits[0]['content']}")
        return "\n".join(lines)

    def _order_reply(self, result: dict) -> str:
        order = result.get("order")
        if not order:
            return "暂未查到您的订单。请提供订单号，例如 O20260617001，我可以继续帮您查询。"
        latest = order["logistics"][-1]["status"] if order.get("logistics") else order["status"]
        return (
            f"订单 {order['order_id']} 当前状态：{order['status']}。"
            f"最新物流：{latest}。预计送达时间：{order['delivery_eta']}。"
        )

    def _refund_reply(self, result: dict) -> str:
        if not result.get("accepted"):
            return result.get("message", "暂时无法发起退款，请补充订单号或转人工处理。")
        ticket = result["ticket"]
        return f"已为您登记售后/退款申请，工单号 {ticket['ticket_id']}。请保留商品照片，客服核实后会处理退款或补偿。"

    def _coupon_reply(self, result: dict) -> str:
        coupons = result["coupons"]
        if not coupons:
            return "当前账号暂无可用优惠券。"
        lines = ["您当前可用优惠券："]
        for coupon in coupons:
            lines.append(f"- {coupon['name']}，适用范围：{coupon['scope']}，有效期至{coupon['valid_until']}")
        return "\n".join(lines)

    def _knowledge_reply(self, hits: list[dict]) -> str:
        if not hits:
            return "我暂时没有找到匹配答案，已建议您补充更多信息或转人工客服。"
        return f"根据知识库：{hits[0]['content']}"


agent = CustomerServiceAgent()
