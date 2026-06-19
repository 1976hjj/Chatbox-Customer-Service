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
python -m pip install -r requirements.txt
python scripts\setup_vector_store.py
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
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

## 前端 React 聊天界面

前端项目放在 `E:\codex\Skill\Customer-Service-Chat-FrontEnd\`，默认连接本地后端 `http://127.0.0.1:8000`。

### 启动后端

```powershell
$env:LLM_PROVIDER="zhipu"
$env:LLM_MODEL="glm-4.7-flash"
$env:LLM_API_KEY="你的智谱 API Key"
$env:LLM_MAX_TOKENS="2048"
$env:LLM_TIMEOUT="180"
py -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 启动前端

```powershell
cd E:\codex\Skill\Customer-Service-Chat-FrontEnd
npm.cmd install
npm.cmd run dev
```

打开：

- http://127.0.0.1:5173

如果后端地址不是 `127.0.0.1:8000`，可以这样指定：

```powershell
$env:VITE_API_BASE="http://127.0.0.1:8000"
npm.cmd run dev
```

界面会显示：

- 客服聊天窗口
- 快捷测试问题
- 后端连接状态
- `route`、`intent`、`tool_calls`、`react_steps` 等 Agent 调试信息

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

### 1.2 商品推荐优化示例

商品库有货时，Agent 会先查商品，再优先连接本地知识库/RAG 生成介绍；如果 RAG 不足，再让大模型做简短销售推荐：

```json
{
  "session_id": "s-sales-001",
  "user_id": "u1001",
  "message": "苹果有没有货，给我推荐一下"
}
```

商品库没有命中时，Agent 不会编造库存，会说明当前没有查到，并基于现有生鲜商品做简短替代推荐：

```json
{
  "session_id": "s-sales-002",
  "user_id": "u1001",
  "message": "能否买高达，介绍一下高达这个玩具"
}
```

```json
{
  "session_id": "s-sales-003",
  "user_id": "u1001",
  "message": "能否买避孕套"
}
```

前端右侧 Agent 面板会展示 `llm_calls`，可以直接看到每次 LLM 调用的 request、raw_response 和解析结果。

例如“介绍一下阿克苏苹果”会优先命中本地知识库 `kb_007`，再生成商品介绍。

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

## 避免 WinError 10013

这个错误通常不是代码逻辑问题，而是 `8000` 或 `5173` 端口已经被之前启动的进程占用了。推荐以后用脚本启动，它会先检查端口，已经启动就直接提示，不会重复绑定端口。

启动后端：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_backend.ps1
```

如果想把后端改到其他端口：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_backend.ps1 -Port 8010
```

停止本项目常用开发端口：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop_dev_ports.ps1
```

如果只想手动查端口：

```powershell
netstat -ano | findstr :8000
netstat -ano | findstr :5173
```

## 代码分层速览

后端入口仍然是 `app/api/routes.py`，聊天请求会进入 `app/services/agent.py` 的
`CustomerServiceAgent.run()`。现在 `agent.py` 主要保留主流程编排：

```text
message
  -> normalize_message
  -> analyze_sentiment
  -> classify_intent
  -> route_intent
  -> plan_next_action
  -> call_tool / search_knowledge
  -> build reply
```

几个核心概念可以按层理解：

- 业务决策字段：`intent`、`confidence`、`sentiment`、`route`
  - `app/services/intent_classifier.py`：规则识别 + LLM 意图融合。
  - `app/services/sentiment.py`：情绪判断。
  - `app/agents/router.py`：把 intent/sentiment 映射成 `order`、`refund`、`product_rag`、`human`、`llm` 等 route。
- 执行字段：`action`、`tool_calls`、`observations`
  - `app/services/llm_service.py`：`plan_next_action()` 让 planner 决定下一步 action。
  - `app/agents/executor.py`：清洗 action 参数、调用工具、生成 Observation 摘要。
  - `app/services/tools.py`：真实工具注册表，比如查商品、查订单、退款、转人工、RAG 检索。
- 调试字段：`thought`、`react_steps`、`llm_calls`
  - `react_steps` 在 `agent.py` 中记录 Thought / Action / Observation / Final。
  - `llm_calls` 来自 `LLMService._record_call()`，记录每次 LLM 的输入、原始输出和解析结果。
- 最终输出字段：`reply`、`citations`、`handler_type`
  - `reply` 仍由 `agent.py` 根据工具结果统一组装。
  - `citations` 来自 `search_knowledge` 的 RAG 命中文档。
  - `handler_type` 通常来自 LLM 意图结果，用于提示 route 是否应走 tool、rag、transfer 或 llm。

`app/agents/state.py` 只是把这些字段按层列成一个轻量数据结构，帮助阅读和后续重构；当前 API 仍保持原来的 dict 返回格式。
