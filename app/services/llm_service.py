import json
import re
import time
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
        self.retry_count = settings.llm_retry_count
        self.retry_delay = settings.llm_retry_delay
        self.calls: list[dict[str, Any]] = []

    def reset_trace(self) -> None:
        self.calls = []

    def analyze_intent(self, text: str, rule_intent: dict | None = None) -> dict[str, Any]:
        if self.provider == "mock" or not self.api_key:
            raw = mock_intent_response(text, rule_intent)
        else:
            prompt = self._intent_prompt(text, rule_intent)
            try:
                raw = self.generate(prompt, temperature=0.1)
            except httpx.HTTPError:
                raw = mock_intent_response(text, rule_intent)
        parsed = self._load_json(raw)
        self._record_call("analyze_intent", {"text": text, "rule_intent": rule_intent}, raw, parsed)
        return parsed

    def generate(self, prompt: str, temperature: float | None = None, max_tokens: int | None = None) -> str:
        if self.provider == "mock" or not self.api_key:
            raw = mock_chat_response(prompt)
            self._record_call("generate", {"prompt": prompt}, raw)
            return raw

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
        raw = self._post_chat_completion(headers, payload)
        self._record_call("generate", {"prompt": prompt}, raw)
        return raw

    def chat(self, messages: list[dict[str, str]]) -> str:
        if self.provider == "mock" or not self.api_key:
            raw = mock_chat_response("", messages)
            self._record_call("chat", {"messages": messages}, raw)
            return raw

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
        raw = self._post_chat_completion(headers, payload)
        self._record_call("chat", {"messages": messages}, raw)
        return raw

    def plan_next_action(self, context: dict[str, Any]) -> dict[str, Any]:
        if self.provider == "mock" or not self.api_key:
            raw = mock_react_action(context)
        else:
            try:
                raw = self.generate(self._react_prompt(context), temperature=0.1)
            except httpx.HTTPError:
                raw = mock_react_action(context)
        parsed = self._load_json(raw)
        self._record_call("plan_next_action", {"context": context}, raw, parsed)
        return parsed

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

    def _post_chat_completion(self, headers: dict[str, str], payload: dict[str, Any]) -> str:
        last_error: httpx.HTTPStatusError | None = None
        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(self.retry_count + 1):
                response = client.post(f"{self.api_base}/chat/completions", headers=headers, json=payload)
                if response.status_code != 429:
                    response.raise_for_status()
                    return self._message_text(response.json())

                last_error = httpx.HTTPStatusError(
                    f"HTTP 429 Too Many Requests: {response.text}",
                    request=response.request,
                    response=response,
                )
                self._record_call(
                    "rate_limited",
                    {
                        "model": payload.get("model"),
                        "attempt": attempt + 1,
                        "retry_after": response.headers.get("Retry-After"),
                    },
                    response.text,
                    {"status_code": 429},
                )
                if attempt >= self.retry_count:
                    break

                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else self.retry_delay * (attempt + 1)
                time.sleep(delay)

        if last_error:
            raise last_error
        raise RuntimeError("LLM request failed without response")

    def _record_call(self, task: str, request: dict[str, Any], raw_response: str, parsed_response: Any = None) -> None:
        self.calls.append(
            {
                "task": task,
                "provider": self.provider,
                "model": self.model,
                "request": request,
                "raw_response": raw_response,
                "parsed_response": parsed_response,
            }
        )

    def _intent_prompt(self, text: str, rule_intent: dict | None) -> str:
        return f"""
请识别生鲜电商客服用户意图，只返回 JSON。

可选 intent_code:
product_inquiry, order_query, refund_request, coupon, complaint, human_agent, greeting, fallback

如果用户问“能否买某物、有没有某物、介绍某商品、库存、价格、推荐”，优先识别为 product_inquiry。
请尽量在 slots.product 中放入用户提到的商品名，例如“高达”“避孕套”“苹果”。

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
- recommend_product(product)
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
