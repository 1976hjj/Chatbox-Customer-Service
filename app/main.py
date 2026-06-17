from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.services.vector_store import vector_store


# FastAPI 应用入口：这里负责创建服务、加载启动资源、挂载接口路由。
settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0")


@app.on_event("startup")
def startup() -> None:
    # 服务启动时先准备知识库索引，后面的搜索接口和客服 Agent 才能直接查询。
    vector_store.load_or_build()


@app.get("/health")
def health() -> dict:
    # 健康检查接口：返回 ok 说明应用进程已经正常启动。
    return {"status": "ok", "app": settings.app_name}


# 把 app/api/routes.py 里的业务接口统一挂到 /api/v1 前缀下。
app.include_router(router, prefix=settings.api_prefix)
