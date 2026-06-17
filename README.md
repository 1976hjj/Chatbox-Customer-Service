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
