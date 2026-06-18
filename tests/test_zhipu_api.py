import os
import unittest

import httpx

from app.core.config import get_settings
from app.services.llm_service import LLMService


@unittest.skipUnless(
    os.getenv("RUN_ZHIPU_API_TEST") == "1",
    "Set RUN_ZHIPU_API_TEST=1 to call the real Zhipu API.",
)
class ZhipuApiTestCase(unittest.TestCase):
    def test_glm_47_flash_salmon_json(self):
        settings = get_settings()
        self.assertEqual(settings.llm_provider, "zhipu")
        self.assertTrue(settings.llm_api_key)
        self.assertEqual(settings.llm_model, "glm-4.7")

        llm = LLMService()
        llm.retry_count = 0
        prompt = """
请介绍一下良心匠人馄饨，并且只返回 JSON，不要 Markdown，不要解释，尽量简短控制在100tokens。
JSON 字段要求：
{
  "product": "良心匠人馄饨",
  "summary": "一句话简介",
  "selling_points": ["卖点1", "卖点2", "卖点3"],
  "storage_advice": "保存建议",
  "suitable_scenes": ["场景1", "场景2"]
}
"""

        try:
            raw = llm.generate(prompt, temperature=0.1, max_tokens=128)
        except httpx.HTTPStatusError as exc:
            print("\nZHIPU_HTTP_STATUS:", exc.response.status_code)
            print("ZHIPU_RESPONSE_BODY:", exc.response.text)
            print("ZHIPU_RETRY_AFTER:", exc.response.headers.get("Retry-After"))
            raise

        print("\nZHIPU_RAW_RESPONSE:")
        print(raw)

        parsed = llm._load_json(raw)
        print("\nZHIPU_PARSED_JSON:")
        print(parsed)

        self.assertTrue(parsed.get("summary"))
        self.assertIsInstance(parsed.get("selling_points"), list)
        self.assertTrue(parsed.get("storage_advice"))


if __name__ == "__main__":
    unittest.main()
