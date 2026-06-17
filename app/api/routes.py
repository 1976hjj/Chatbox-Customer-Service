from fastapi import APIRouter

from app.core.schemas import AgentRunRequest, ChatRequest, ChatResponse, IntentRequest, IntentResult, SearchHit, SearchRequest
from app.services.agent import agent
from app.services.intent_classifier import classify_intent
from app.services.preprocessor import normalize_message
from app.services.vector_store import vector_store

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    return agent.run(payload.session_id, payload.user_id, payload.message)


@router.post("/intent/classify", response_model=IntentResult)
def classify(payload: IntentRequest):
    return classify_intent(normalize_message(payload.message))


@router.post("/search", response_model=list[SearchHit])
def search(payload: SearchRequest):
    return vector_store.search(payload.query, payload.top_k)


@router.post("/agent/run", response_model=ChatResponse)
def run_agent(payload: AgentRunRequest):
    return agent.run(payload.session_id, payload.user_id, payload.message)
