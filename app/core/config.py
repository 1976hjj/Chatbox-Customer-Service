from functools import lru_cache
import os
from pathlib import Path

from pydantic import BaseModel


# 全局配置模型：集中保存应用名、接口前缀、数据目录和 Agent 步数等基础参数。
class Settings(BaseModel):
    app_name: str = "Chatbox Customer Service"
    api_prefix: str = "/api/v1"
    data_dir: Path = Path(__file__).resolve().parents[1] / "data"
    vector_index_path: Path = Path(__file__).resolve().parents[1] / "data" / "vector_index.json"
    max_agent_steps: int = 4
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock")
    llm_model: str = os.getenv("LLM_MODEL", "glm-4.7-flash")
    llm_api_base: str = os.getenv("LLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "180"))


@lru_cache
def get_settings() -> Settings:
    # 用缓存保证全项目拿到同一份配置，同时自动创建本地数据目录。
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
