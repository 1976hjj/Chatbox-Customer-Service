import json
import math
from collections import Counter
from pathlib import Path

from app.core.config import get_settings
from app.data.knowledge import KNOWLEDGE_DOCS


class LocalVectorStore:
    """Small persistent vector store for local development and tests."""

    def __init__(self, index_path: Path | None = None):
        self.index_path = index_path or get_settings().vector_index_path
        self.documents: list[dict] = []

    def build(self, docs: list[dict] | None = None) -> None:
        source_docs = docs or KNOWLEDGE_DOCS
        self.documents = []
        for doc in source_docs:
            text = f"{doc['title']} {doc['content']}"
            self.documents.append({**doc, "vector": self._embed(text)})
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(self.documents, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_or_build(self) -> None:
        if self.index_path.exists():
            self.documents = json.loads(self.index_path.read_text(encoding="utf-8"))
        else:
            self.build()

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        self.load_or_build()
        query_vector = self._embed(query)
        hits = []
        for doc in self.documents:
            score = self._cosine(query_vector, doc["vector"])
            hits.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {}),
                    "score": round(score, 4),
                }
            )
        return sorted(hits, key=lambda item: item["score"], reverse=True)[:top_k]

    def _embed(self, text: str) -> dict[str, float]:
        tokens = self._tokenize(text)
        counts = Counter(tokens)
        norm = math.sqrt(sum(value * value for value in counts.values())) or 1.0
        return {key: value / norm for key, value in counts.items()}

    def _tokenize(self, text: str) -> list[str]:
        cleaned = text.lower().replace(" ", "")
        tokens = []
        for size in (1, 2, 3):
            tokens.extend(cleaned[index : index + size] for index in range(max(0, len(cleaned) - size + 1)))
        return tokens

    def _cosine(self, left: dict[str, float], right: dict[str, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(key, 0.0) for key, value in left.items())


vector_store = LocalVectorStore()
