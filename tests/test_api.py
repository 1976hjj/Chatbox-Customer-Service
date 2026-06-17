import unittest

from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


class ApiTestCase(unittest.TestCase):
    def test_health(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_intent_order_query(self):
        response = client.post("/api/v1/intent/classify", json={"message": "我的订单 O20260617001 到哪了"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["intent_id"], "intent_100")
        self.assertEqual(data["slots"]["order_id"], "O20260617001")

    def test_search_can_retrieve_after_sale_policy(self):
        response = client.post("/api/v1/search", json={"query": "坏果怎么赔付", "top_k": 2})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 2)
        self.assertGreater(data[0]["score"], 0)

    def test_chat_order_query(self):
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
        response = client.post(
            "/api/v1/chat",
            json={"session_id": "s-test", "user_id": "u1001", "message": "牛油果有没有货，怎么保存"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["route"], "product_rag")
        self.assertTrue(data["citations"])

    def test_negative_complaint_transfer_to_human(self):
        response = client.post(
            "/api/v1/chat",
            json={"session_id": "s-test", "user_id": "u1001", "message": "我要投诉，苹果坏了，太生气了"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["route"], "human")
        self.assertEqual(data["tool_calls"][0]["name"], "transfer_human")


if __name__ == "__main__":
    unittest.main()
