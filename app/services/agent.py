from datetime import datetime
from typing import Any

from app.core.config import get_settings
from app.repositories import mock_db
from app.services.intent_classifier import classify_intent
from app.services.llm_service import LLMService
from app.services.preprocessor import normalize_message
from app.services.sentiment import analyze_sentiment
from app.services.tools import TOOL_REGISTRY


class CustomerServiceAgent:
    """Customer-service Agent with an LLM-driven ReAct planning loop."""

    def __init__(self) -> None:
        self.llm_service = LLMService()
        self.max_steps = get_settings().max_agent_steps

    def run(self, session_id: str, user_id: str, message: str) -> dict:
        self.llm_service.reset_trace()
        normalized = normalize_message(message)
        sentiment = analyze_sentiment(normalized)
        intent = classify_intent(normalized, self.llm_service)

        tool_calls: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        citations: list[dict[str, Any]] = []
        react_steps = [
            f"Thought: normalized message = {normalized}",
            f"Thought: intent = {intent['intent_id']} from {intent['slots'].get('intent_source', 'rule')}",
            f"Thought: sentiment = {sentiment['label']}({sentiment['score']})",
        ]

        reply = ""
        route = self._route(intent, sentiment)

        for step in range(1, self.max_steps + 1):
            decision = self.llm_service.plan_next_action(
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "message": normalized,
                    "intent": intent,
                    "sentiment": sentiment,
                    "route_hint": route,
                "available_tools": sorted(TOOL_REGISTRY.keys()),
                    "observations": observations,
                }
            )
            action = decision.get("action", "final")
            action_input = decision.get("action_input") or {}
            react_steps.append(f"Thought[{step}]: {decision.get('thought', '')}")
            react_steps.append(f"Action[{step}]: {action} {action_input}")

            if action == "final":
                reply = decision.get("final_answer") or self._reply_from_observations(route, tool_calls, citations, normalized)
                react_steps.append(f"Final[{step}]: reply ready")
                break

            if action not in TOOL_REGISTRY:
                reply = self.llm_service.generate("Unknown tool requested. Please provide a short fallback reply.")
                react_steps.append(f"Observation[{step}]: unknown action {action}")
                react_steps.append(f"Final[{step}]: fallback reply ready")
                break

            safe_input = self._normalize_action_input(action, action_input, session_id, user_id, normalized, intent)
            tool_call = self._call(action, **safe_input)
            tool_calls.append(tool_call)
            observation = {"action": action, "result": tool_call["result"]}
            observations.append(observation)
            react_steps.append(f"Observation[{step}]: {self._summarize_observation(action, tool_call['result'])}")

            if action == "search_knowledge":
                citations = tool_call["result"].get("hits", [])
                if route in {"product_rag", "knowledge"}:
                    reply = self._reply_from_observations(route, tool_calls, citations, normalized)
                    react_steps.append(f"Final[{step}]: reply built after knowledge search")
                    break
            elif action in {"query_order", "refund_order", "list_coupon", "transfer_human", "recommend_product"}:
                reply = self._reply_from_observations(route, tool_calls, citations, normalized)
                react_steps.append(f"Final[{step}]: reply built after {action}")
                break
        else:
            reply = self._reply_from_observations(route, tool_calls, citations, normalized)
            react_steps.append("Final: max steps reached, reply built from observations")

        record = {
            "session_id": session_id,
            "user_id": user_id,
            "message": message,
            "normalized_message": normalized,
            "intent": intent,
            "sentiment": sentiment,
            "reply": reply,
            "route": route,
            "react_steps": react_steps,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        mock_db.save_chat(record)
        return {
            **record,
            "tool_calls": tool_calls,
            "citations": citations,
            "llm": {
                "provider": self.llm_service.provider,
                "model": self.llm_service.model,
                "intent_source": intent["slots"].get("intent_source", "rule"),
                "llm_intent_code": intent["slots"].get("llm_intent_code"),
                "planner": "react_loop",
            },
            "llm_calls": self.llm_service.calls,
        }

    def _route(self, intent: dict, sentiment: dict) -> str:
        handler_type = intent["slots"].get("handler_type")
        if handler_type == "transfer":
            return "human"
        if handler_type == "llm":
            return "llm"
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

    def _normalize_action_input(
        self,
        action: str,
        action_input: dict[str, Any],
        session_id: str,
        user_id: str,
        message: str,
        intent: dict,
    ) -> dict[str, Any]:
        slots = intent.get("slots", {})
        if action == "query_product":
            return {"product": slots.get("product") or action_input.get("product")}
        if action == "search_knowledge":
            return {"query": action_input.get("query") or message, "top_k": action_input.get("top_k", 3)}
        if action == "query_order":
            return {"order_id": slots.get("order_id") or action_input.get("order_id"), "user_id": user_id}
        if action == "refund_order":
            return {"order_id": slots.get("order_id") or action_input.get("order_id"), "user_id": user_id, "reason": action_input.get("reason") or message}
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

    def _call(self, name: str, **kwargs) -> dict:
        result = TOOL_REGISTRY[name](**kwargs)
        return {"name": name, "arguments": kwargs, "result": result}

    def _summarize_observation(self, action: str, result: dict) -> str:
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

    def _reply_from_observations(self, route: str, tool_calls: list[dict], citations: list[dict], message: str) -> str:
        results = {call["name"]: call["result"] for call in tool_calls}
        if "refund_order" in results:
            return self._refund_reply(results["refund_order"])
        if "query_order" in results:
            return self._order_reply(results["query_order"])
        if "list_coupon" in results:
            return self._coupon_reply(results["list_coupon"])
        if "transfer_human" in results:
            return "已为您创建人工客服工单，客服会优先处理。您也可以继续补充订单号、商品照片或问题细节。"
        if "query_product" in results:
            product_call = next((call for call in tool_calls if call["name"] == "query_product"), tool_calls[0])
            product = product_call["arguments"].get("product")
            if not results["query_product"].get("products") and "recommend_product" in results:
                return self._unavailable_product_reply(message, product, results["recommend_product"])
            return self._product_reply(message, product, results["query_product"], citations)
        if "recommend_product" in results:
            return self._unavailable_product_reply(message, None, results["recommend_product"])
        if "search_knowledge" in results:
            return self._knowledge_reply(citations)
        if route == "llm":
            return self.llm_service.chat(
                [
                    {"role": "system", "content": "你是生鲜电商智能客服，回答要简洁、友好、可执行。"},
                    {"role": "user", "content": "请给用户一个简短回复。"},
                ]
            )
        return "我已收到您的问题，会继续为您处理。"

    def _product_reply(self, message: str, product: str | None, product_result: dict, hits: list[dict]) -> str:
        products = product_result["products"]
        if not products:
            return f"暂时没有查到和{product or '该商品'}相关的在售商品。您可以换个关键词，我也可以为您推荐相近商品。"
        rag_hits = self._relevant_rag_hits(hits)
        if rag_hits:
            return self._product_rag_reply(message, product, products, rag_hits)
        context = {
            "user_message": message,
            "requested_product": product,
            "products": products[:3],
            "knowledge": [hit["content"] for hit in hits[:2]],
        }
        prompt = f"""
你是生鲜电商客服导购。请基于以下 JSON 回复用户，要求：
1. 只推荐 JSON 中真实存在的商品，不要编造库存或商品。
2. 如果有库存，给出 1-3 个推荐点：价格、库存、新鲜/保存/适用场景。
3. 如果知识库有内容，可自然补充一句。
4. 回复控制在 120 字以内，语气自然，有一点销售引导。

商品推荐上下文:
{context}
"""
        return self.llm_service.generate(prompt)

    def _product_rag_reply(self, message: str, product: str | None, products: list[dict], hits: list[dict]) -> str:
        context = {
            "user_message": message,
            "requested_product": product,
            "products": products[:3],
            "rag_knowledge": [
                {
                    "title": hit["title"],
                    "content": hit["content"],
                    "score": hit["score"],
                    "metadata": hit.get("metadata", {}),
                }
                for hit in hits[:3]
            ],
        }
        prompt = f"""
你是生鲜电商客服导购。请优先基于本地 RAG 知识库回答用户，不要只说“库存充足”。
要求：
1. 先介绍商品特点/口感/规格/保存建议，内容必须来自 rag_knowledge 或 products。
2. 可以补充价格和库存，但不要编造没有给出的信息。
3. 如果用户问“介绍”，回答要像商品介绍；如果问“能不能买”，再强调是否在售。
4. 回复控制在 160 字以内，自然、有销售引导。

RAG 商品上下文:
{context}
"""
        return self.llm_service.generate(prompt)

    def _unavailable_product_reply(self, message: str, product: str | None, recommend_result: dict) -> str:
        recommendations = recommend_result.get("recommendations", [])
        context = {
            "user_message": message,
            "requested_product": product,
            "recommendations": recommendations[:3],
        }
        prompt = f"""
你是生鲜电商客服。用户问的商品当前本地商品库没有命中，请基于以下 JSON 回复：
1. 明确说明“当前没有查到 {product or '该商品'} 在售”。
2. 不要装作有货，不要介绍平台没有的商品细节。
3. 如果有 recommendations，推荐 1-3 个现有可售替代商品，带价格/库存亮点。
4. 如果用户问的是明显非生鲜品类，例如高达、避孕套，也要礼貌说明当前主要售卖生鲜食品，可推荐水果、鸡蛋、水产等。
5. 回复控制在 120 字以内。

替代推荐上下文:
{context}
"""
        return self.llm_service.generate(prompt)

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
            return self.llm_service.generate("用户问题没有命中知识库，请给出简短客服兜底回复。")
        context = "\n".join(hit["content"] for hit in hits[:3])
        return self.llm_service.generate(f"知识库内容:\n{context}\n\n请基于知识库回答用户问题。")

    def _relevant_rag_hits(self, hits: list[dict]) -> list[dict]:
        return [hit for hit in hits if hit.get("score", 0) >= 0.08]


agent = CustomerServiceAgent()
