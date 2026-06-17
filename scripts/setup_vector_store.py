from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.vector_store import vector_store  # noqa: E402


if __name__ == "__main__":
    vector_store.build()
    print(f"Vector index ready: {vector_store.index_path}")
