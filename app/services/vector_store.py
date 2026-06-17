import json
import math
from collections import Counter
from pathlib import Path

from app.core.config import get_settings
from app.data.knowledge import KNOWLEDGE_DOCS


# 本地知识库模块：把文档转成简单向量并保存到 JSON，提供相似度搜索。
class LocalVectorStore:
    """Small persistent vector store for local development and tests."""

    def __init__(self, index_path: Path | None = None):
        # 初始化时确定索引文件路径，并准备一个内存文档列表。
        self.index_path = index_path or get_settings().vector_index_path
        self.documents: list[dict] = []

    def build(self, docs: list[dict] | None = None) -> None:
        # 从知识文档构建向量索引，并持久化到本地 JSON 文件。
        source_docs = docs or KNOWLEDGE_DOCS
        self.documents = []
        for doc in source_docs:
            text = f"{doc['title']} {doc['content']}"
            self.documents.append({**doc, "vector": self._embed(text)})
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(self.documents, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_or_build(self) -> None:
        # 优先加载已存在的索引文件；没有文件时自动重新构建。
        if self.index_path.exists():
            self.documents = json.loads(self.index_path.read_text(encoding="utf-8"))
        else:
            self.build()

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        # 把查询文本向量化后，与每篇文档算相似度并返回最高的结果。
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
        # 用字符 n-gram 计数生成归一化向量，适合没有外部模型的本地演示。
        tokens = self._tokenize(text)
        counts = Counter(tokens)
        norm = math.sqrt(sum(value * value for value in counts.values())) or 1.0
        return {key: value / norm for key, value in counts.items()}

    def _tokenize(self, text: str) -> list[str]:
        # 把文本拆成 1、2、3 字符片段，让中文短词也能参与匹配。
        cleaned = text.lower().replace(" ", "")
        tokens = []
        for size in (1, 2, 3):
            tokens.extend(cleaned[index : index + size] for index in range(max(0, len(cleaned) - size + 1)))
        return tokens

    def _cosine(self, left: dict[str, float], right: dict[str, float]) -> float:
        # 计算两个稀疏向量的点积，值越高说明 query 和文档越相似。
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(key, 0.0) for key, value in left.items())


# 全项目共用一个本地知识库实例，避免每个接口重复创建。
vector_store = LocalVectorStore()
