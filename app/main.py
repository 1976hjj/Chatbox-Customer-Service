from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.services.vector_store import vector_store


settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0")


@app.on_event("startup")
def startup() -> None:
    vector_store.load_or_build()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "app": settings.app_name}


app.include_router(router, prefix=settings.api_prefix)
