import unittest
import os

os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("LLM_API_KEY", "")

from fastapi.testclient import TestClient
from app.main import app


# API 测试模块：用 TestClient 在本地直接调用 FastAPI 接口，不需要启动服务器。
client = TestClient(app)


class ApiTestCase(unittest.TestCase):
    def test_health(self):
        # 验证健康检查接口能返回 ok。
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_intent_order_query(self):
        # 验证带订单号的消息能识别为订单查询，并抽取出订单号。
        response = client.post("/api/v1/intent/classify", json={"message": "我的订单 O20260617001 到哪了"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["intent_id"], "intent_100")
        self.assertEqual(data["slots"]["order_id"], "O20260617001")

    def test_search_can_retrieve_after_sale_policy(self):
        # 验证知识库搜索能返回指定数量的相似文档。
        response = client.post("/api/v1/search", json={"query": "坏果怎么赔付", "top_k": 2})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 2)
        self.assertGreater(data[0]["score"], 0)

    def test_chat_order_query(self):
        # 验证聊天 Agent 遇到订单问题时会走订单查询工具。
        response = client.post(
            "/api/v1/chat",
            json={"session_id": "s-test", "user_id": "u1001", "message": "我的订单 O20260617001 到哪了"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["route"], "order")
        self.assertIn("配送中", data["reply"])
        self.assertEqual(data["tool_calls"][0]["name"], "query_order")

    def test_chat_product_rag(self):
        # 验证商品咨询会走商品查询加知识库补充说明的流程。
        response = client.post(
            "/api/v1/chat",
            json={"session_id": "s-test", "user_id": "u1001", "message": "牛油果有没有货，怎么保存"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["route"], "product_rag")
        self.assertTrue(data["citations"])

    def test_negative_complaint_transfer_to_human(self):
        # 验证负面投诉会被路由到人工客服工单。
        response = client.post(
            "/api/v1/chat",
            json={"session_id": "s-test", "user_id": "u1001", "message": "我要投诉，苹果坏了，太生气了"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["route"], "human")
        self.assertEqual(data["tool_calls"][0]["name"], "transfer_human")


    def test_llm_mock_fuses_refund_intent(self):
        message = "\u8ba2\u5355 O20260617001 \u7684\u82f9\u679c\u574f\u4e86\uff0c\u5e2e\u6211\u7533\u8bf7\u552e\u540e\u8d54\u4ed8"
        response = client.post(
            "/api/v1/chat",
            json={"session_id": "s-test", "user_id": "u1001", "message": message},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["route"], "refund")
        self.assertEqual(data["intent"]["slots"]["intent_source"], "llm")
        self.assertEqual(data["tool_calls"][0]["name"], "refund_order")
        self.assertTrue(data["react_steps"])

    def test_llm_mock_direct_reply_for_greeting(self):
        message = "\u4f60\u597d\uff0c\u5728\u5417"
        response = client.post(
            "/api/v1/chat",
            json={"session_id": "s-test", "user_id": "u1001", "message": message},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["route"], "llm")
        self.assertEqual(data["llm"]["provider"], "mock")
        self.assertIn("您好", data["reply"])


    def test_react_planner_can_choose_multiple_actions(self):
        message = "\u82f9\u679c\u6709\u6ca1\u6709\u8d27\uff0c\u600e\u4e48\u4fdd\u5b58"
        response = client.post(
            "/api/v1/chat",
            json={"session_id": "s-test", "user_id": "u1001", "message": message},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        actions = [call["name"] for call in data["tool_calls"]]
        self.assertEqual(actions, ["query_product", "search_knowledge"])
        self.assertEqual(data["llm"]["planner"], "react_loop")
        self.assertTrue(any("Action[1]: query_product" in step for step in data["react_steps"]))
        self.assertTrue(any("Action[2]: search_knowledge" in step for step in data["react_steps"]))

    def test_missing_product_gets_alternative_recommendations(self):
        message = "\u80fd\u5426\u4e70\u9ad8\u8fbe\uff0c\u4ecb\u7ecd\u4e00\u4e0b\u9ad8\u8fbe\u8fd9\u4e2a\u73a9\u5177"
        response = client.post(
            "/api/v1/chat",
            json={"session_id": "s-test", "user_id": "u1001", "message": message},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        actions = [call["name"] for call in data["tool_calls"]]
        self.assertIn("query_product", actions)
        self.assertIn("recommend_product", actions)
        self.assertIn("\u6ca1\u6709\u67e5\u5230", data["reply"])

    def test_available_product_gets_llm_sales_reply(self):
        message = "\u82f9\u679c\u6709\u6ca1\u6709\u8d27\uff0c\u7ed9\u6211\u63a8\u8350\u4e00\u4e0b"
        response = client.post(
            "/api/v1/chat",
            json={"session_id": "s-test", "user_id": "u1001", "message": message},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("query_product", [call["name"] for call in data["tool_calls"]])
        self.assertNotIn("\u6682\u65f6\u6ca1\u6709\u67e5\u5230", data["reply"])
        self.assertTrue(data["reply"])

    def test_product_intro_prefers_local_rag_and_exposes_llm_calls(self):
        message = "\u4ecb\u7ecd\u4e00\u4e0b\u963f\u514b\u82cf\u82f9\u679c"
        response = client.post(
            "/api/v1/chat",
            json={"session_id": "s-test", "user_id": "u1001", "message": message},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["route"], "product_rag")
        self.assertIn("search_knowledge", [call["name"] for call in data["tool_calls"]])
        self.assertTrue(any(hit["id"] == "kb_007" for hit in data["citations"]))
        self.assertIn("\u963f\u514b\u82cf\u82f9\u679c", data["reply"])
        self.assertTrue(data["llm_calls"])
        self.assertTrue(any(call["task"] == "generate" for call in data["llm_calls"]))


if __name__ == "__main__":
    unittest.main()
