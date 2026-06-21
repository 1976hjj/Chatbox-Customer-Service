import os
import unittest

os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("LLM_API_KEY", "")

from app.services.agent import CustomerServiceAgent


class FlexibleRouteActionTestCase(unittest.TestCase):
    def test_coupon_route_can_search_knowledge_before_final_reply(self):
        agent = CustomerServiceAgent()
        planned_actions = []

        def planner(context: dict) -> dict:
            planned_actions.append((context["allowed_actions"], context["observations"]))
            if not context["observations"]:
                return {"thought": "Query the user's coupons first.", "action": "list_coupon", "action_input": {}}
            if len(context["observations"]) == 1:
                return {
                    "thought": "Use knowledge retrieval to explain the coupon rules.",
                    "action": "search_knowledge",
                    "action_input": {"query": "coupon rules", "top_k": 2},
                }
            return {
                "thought": "The coupon and policy data are sufficient.",
                "action": "final",
                "action_input": {},
                "final_answer": "You have available coupons. The matching rule has been checked.",
            }

        original_planner = agent.llm_service.plan_next_action
        agent.llm_service.plan_next_action = planner
        try:
            result = agent.run("s-flexible", "u1001", "\u6211\u6709\u4ec0\u4e48\u4f18\u60e0\u5238\u53ef\u4ee5\u7528\uff1f")
        finally:
            agent.llm_service.plan_next_action = original_planner

        self.assertEqual(result["route"], "coupon")
        self.assertEqual([call["name"] for call in result["tool_calls"]], ["list_coupon", "search_knowledge"])
        self.assertEqual(planned_actions[0][0], ["list_coupon"])
        self.assertEqual(planned_actions[1][0], ["search_knowledge", "final"])
        self.assertEqual(planned_actions[2][0], ["final"])
        self.assertEqual(result["reply"], "You have available coupons. The matching rule has been checked.")


if __name__ == "__main__":
    unittest.main()
