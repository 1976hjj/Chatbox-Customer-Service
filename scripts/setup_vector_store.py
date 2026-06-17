from pathlib import Path
import sys

# 脚本入口：手动运行时用来提前生成本地知识库索引文件。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.vector_store import vector_store  # noqa: E402


if __name__ == "__main__":
    # 直接执行该脚本时，重建索引并打印索引文件位置。
    vector_store.build()
    print(f"Vector index ready: {vector_store.index_path}")
