from fastapi import APIRouter

from app.core.schemas import AgentRunRequest, ChatRequest, ChatResponse, IntentRequest, IntentResult, SearchHit, SearchRequest
from app.services.agent import agent
from app.services.intent_classifier import classify_intent
from app.services.preprocessor import normalize_message
from app.services.vector_store import vector_store

# API 路由层：只负责接收 HTTP 请求，并把任务交给对应的服务函数处理。
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    # 聊天入口：把会话、用户和消息交给客服 Agent 生成完整回复。
    return agent.run(payload.session_id, payload.user_id, payload.message)


@router.post("/intent/classify", response_model=IntentResult)
def classify(payload: IntentRequest):
    # 意图识别入口：先清洗用户话术，再判断用户想咨询哪类问题。
    return classify_intent(normalize_message(payload.message))


@router.post("/search", response_model=list[SearchHit])
def search(payload: SearchRequest):
    # 知识库搜索入口：按用户 query 返回相似度最高的几条知识。
    return vector_store.search(payload.query, payload.top_k)


@router.post("/agent/run", response_model=ChatResponse)
def run_agent(payload: AgentRunRequest):
    # Agent 调试入口：功能和 /chat 相同，方便单独测试完整编排流程。
    return agent.run(payload.session_id, payload.user_id, payload.message)
