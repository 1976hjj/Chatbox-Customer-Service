# Chatbox Customer Service

生鲜电商智能客服 Agent 后端原型。当前版本聚焦后端能力，包含：

- FastAPI 接口层：Chat API、Intent API、Agent API、Search API。
- 业务逻辑层：消息预处理、意图分类、情绪分析、Agent 调度。
- AI 服务层：本地 Embedding、RAG 检索、规则化回复生成。
- 数据层：mock 商品、订单、优惠券、用户、工单数据。
- 向量检索：默认自动构建本地 JSON 向量索引，可选 Docker Qdrant 环境。

## 快速启动

```powershell
cd Chatbox-Customer-Service
py -m pip install -r requirements.txt
py scripts\setup_vector_store.py
py -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

打开接口文档：

- Swagger: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## Postman 调用示例

### 1. 对话接口

POST `http://127.0.0.1:8000/api/v1/chat`

```json
{
  "session_id": "s-001",
  "user_id": "u1001",
  "message": "我的订单 O20260617001 到哪了？"
}
```

### 2. 意图识别

POST `http://127.0.0.1:8000/api/v1/intent/classify`

```json
{
  "message": "我想查一下牛油果有没有货"
}
```

### 3. 知识库检索

POST `http://127.0.0.1:8000/api/v1/search`

```json
{
  "query": "生鲜坏果怎么赔付",
  "top_k": 3
}
```

### 4. Agent 直接执行

POST `http://127.0.0.1:8000/api/v1/agent/run`

```json
{
  "session_id": "s-002",
  "user_id": "u1001",
  "message": "苹果坏了，我要退款"
}
```

## 可选 Qdrant 环境

当前代码默认使用本地向量索引，保证无需 Docker 也可运行。后续要接真实 Qdrant，可先启动：

```powershell
docker compose up -d qdrant
```

Qdrant 控制台：

- http://127.0.0.1:6333/dashboard

## 测试

```powershell
py -m unittest discover -s tests -v
```

## LLM mock + ReAct 测试示例

当前默认使用 `mock` LLM，不会真实请求智谱接口。相关代码：

- `app/services/llm_mock.py`：模拟 LLM 的意图识别 JSON 和客服回复。
- `app/services/llm_service.py`：统一 LLM 入口，后续可把 `llm_provider`、`llm_api_key` 改成真实智谱配置。
- `app/services/intent_classifier.py`：先规则识别，再调用 mock LLM，最后融合意图。
- `app/services/agent.py`：返回 `react_steps`，展示 Thought / Action / Observation / Final。

现在的 Agent 已经不是固定 `if route == ...` 后直接调用工具，而是 ReAct planner loop：

1. Agent 把用户消息、意图、情绪、可用工具、历史 Observation 交给 `LLMService.plan_next_action()`。
2. LLM planner 返回 JSON：`thought`、`action`、`action_input`、`final_answer`。
3. Agent 只执行这个 action，并把工具结果作为 Observation 追加回上下文。
4. 下一轮继续让 LLM planner 决定是否调用下一个工具，直到 action 是 `final` 或达到 `max_agent_steps`。

mock planner 在 `app/services/llm_mock.py` 的 `mock_react_action()` 里；真实智谱 planner 的 prompt 在 `app/services/llm_service.py` 的 `_react_prompt()` 里。

### 1. LLM 融合识别售后退款

POST `http://127.0.0.1:8000/api/v1/chat`

```json
{
  "session_id": "s-llm-001",
  "user_id": "u1001",
  "message": "订单 O20260617001 的苹果坏了，帮我申请售后赔付"
}
```

重点看响应里的这些字段：

```json
{
  "route": "refund",
  "intent": {
    "slots": {
      "intent_source": "llm",
      "llm_intent_code": "refund_request"
    }
  },
  "llm": {
    "provider": "mock"
  },
  "react_steps": [
    "Thought: ...",
    "Action: refund_order",
    "Observation: refund tool returned",
    "Final: reply ready"
  ]
}
```

### 1.1 LLM 自己连续选择多个 Action

POST `http://127.0.0.1:8000/api/v1/chat`

```json
{
  "session_id": "s-react-001",
  "user_id": "u1001",
  "message": "苹果有没有货，怎么保存"
}
```

预期工具调用顺序：

```json
{
  "tool_calls": [
    {"name": "query_product"},
    {"name": "search_knowledge"}
  ],
  "llm": {
    "planner": "react_loop"
  },
  "react_steps": [
    "Thought[1]: ...",
    "Action[1]: query_product ...",
    "Observation[1]: products=...",
    "Thought[2]: ...",
    "Action[2]: search_knowledge ...",
    "Observation[2]: hits=...",
    "Thought[3]: ...",
    "Action[3]: final ..."
  ]
}
```

### 2. LLM 直接回答问候

POST `http://127.0.0.1:8000/api/v1/chat`

```json
{
  "session_id": "s-llm-002",
  "user_id": "u1001",
  "message": "你好，在吗"
}
```

预期会走：

```json
{
  "route": "llm",
  "llm": {
    "provider": "mock",
    "llm_intent_code": "greeting"
  }
}
```

### 3. 后续切换智谱的位置

目前 `app/core/config.py` 里默认：

```python
llm_provider = "mock"
llm_model = "glm-4.7-flash"
llm_api_base = "https://open.bigmodel.cn/api/paas/v4"
llm_api_key = ""
```

后续如果要接真实智谱，把 `llm_provider` 改成非 `mock`，并填入 `llm_api_key` 即可复用 `LLMService` 里的 `/chat/completions` 调用结构。

推荐不要把 key 写进代码，直接用环境变量启动：

```powershell
$env:LLM_PROVIDER="zhipu"
$env:LLM_MODEL="glm-4.7-flash"
$env:LLM_API_KEY="你的智谱 API Key"
$env:LLM_MAX_TOKENS="2048"
$env:LLM_TIMEOUT="180"
py -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

真实模型跑通后，响应里的 `llm` 字段会类似：

```json
{
  "provider": "zhipu",
  "model": "glm-4.7-flash",
  "intent_source": "llm",
  "llm_intent_code": "refund_request",
  "planner": "react_loop"
}
```
