import json
import re
from typing import Any

import httpx

from app.core.config import get_settings
from app.services.llm_mock import mock_chat_response, mock_intent_response, mock_react_action


class LLMService:
    """Unified LLM entrypoint; defaults to mock and can later switch to Zhipu."""

    def __init__(self) -> None:
        settings = get_settings()
        self.provider = settings.llm_provider
        self.model = settings.llm_model
        self.api_base = settings.llm_api_base
        self.api_key = settings.llm_api_key
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.timeout = settings.llm_timeout

    def analyze_intent(self, text: str, rule_intent: dict | None = None) -> dict[str, Any]:
        if self.provider == "mock" or not self.api_key:
            raw = mock_intent_response(text, rule_intent)
        else:
            prompt = self._intent_prompt(text, rule_intent)
            try:
                raw = self.generate(prompt, temperature=0.1)
            except httpx.HTTPError:
                raw = mock_intent_response(text, rule_intent)
        return self._load_json(raw)

    def generate(self, prompt: str, temperature: float | None = None, max_tokens: int | None = None) -> str:
        if self.provider == "mock" or not self.api_key:
            return mock_chat_response(prompt)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.api_base}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            return self._message_text(response.json())

    def chat(self, messages: list[dict[str, str]]) -> str:
        if self.provider == "mock" or not self.api_key:
            return mock_chat_response("", messages)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.api_base}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            return self._message_text(response.json())

    def plan_next_action(self, context: dict[str, Any]) -> dict[str, Any]:
        if self.provider == "mock" or not self.api_key:
            raw = mock_react_action(context)
        else:
            try:
                raw = self.generate(self._react_prompt(context), temperature=0.1)
            except httpx.HTTPError:
                raw = mock_react_action(context)
        return self._load_json(raw)

    def _load_json(self, raw: str) -> dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def _message_text(self, payload: dict[str, Any]) -> str:
        message = payload["choices"][0].get("message", {})
        return message.get("content") or message.get("reasoning_content", "")

    def _intent_prompt(self, text: str, rule_intent: dict | None) -> str:
        return f"""
请识别生鲜电商客服用户意图，只返回 JSON。

可选 intent_code:
product_inquiry, order_query, refund_request, coupon, complaint, human_agent, greeting, fallback

用户输入: {text}
规则识别结果: {json.dumps(rule_intent or {}, ensure_ascii=False)}

返回字段:
intent_code, intent_id, intent_name, category, confidence, priority, handler_type, slots
"""

    def _react_prompt(self, context: dict[str, Any]) -> str:
        return f"""
You are a customer-service ReAct planner. Decide exactly one next step.

Available actions:
- query_product(product)
- search_knowledge(query, top_k)
- query_order(order_id, user_id)
- refund_order(order_id, user_id, reason)
- list_coupon(user_id)
- transfer_human(user_id, session_id, message, reason)
- final

Return only JSON:
{{
  "thought": "why this action is next",
  "action": "one action name",
  "action_input": {{}},
  "final_answer": "only when action is final"
}}

Context:
{json.dumps(context, ensure_ascii=False, default=str)}
"""
