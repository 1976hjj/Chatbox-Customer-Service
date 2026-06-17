from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "Chatbox Customer Service"
    api_prefix: str = "/api/v1"
    data_dir: Path = Path(__file__).resolve().parents[1] / "data"
    vector_index_path: Path = Path(__file__).resolve().parents[1] / "data" / "vector_index.json"
    max_agent_steps: int = 4


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
