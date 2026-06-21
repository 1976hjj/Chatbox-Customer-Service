from typing import Any, Callable

from app.repositories import mock_db
from app.services.vector_store import vector_store


# 工具层：把查商品、查订单、退货、转人工、搜知识库等能力封装成 Agent 可调用函数。
ToolFunc = Callable[..., Any]


def query_product(product: str | None = None) -> dict:
    # 查询商品列表；传商品关键词时只返回相关商品。
    products = mock_db.search_products(product)
    return {"products": products[:5]}


def recommend_product(product: str | None = None) -> dict:
    # 按库存从高到低推荐商品，没有关键词命中时就从全量商品里推荐。
    products = mock_db.search_products(product) or mock_db.PRODUCTS
    return {"recommendations": sorted(products, key=lambda item: item["stock"], reverse=True)[:3]}


def query_order(order_id: str | None, user_id: str) -> dict:
    # 查询用户订单；有订单号优先按订单号查，没有则取该用户最近的一单。
    order = mock_db.get_order(order_id, user_id)
    return {"order": order}


def list_coupon(user_id: str) -> dict:
    # 查询用户当前可用优惠券。
    return {"coupons": mock_db.list_coupons(user_id)}


def refund_order(order_id: str | None, user_id: str, reason: str) -> dict:
    # 发起退款：先确认订单存在，存在就创建一个售后工单。
    order = mock_db.get_order(order_id, user_id)
    if not order:
        return {"accepted": False, "message": "未找到可退款订单"}
    ticket = mock_db.create_ticket(user_id, "refund", reason, "refund")
    return {"accepted": True, "order": order, "ticket": ticket}


def transfer_human(user_id: str, session_id: str, message: str, reason: str) -> dict:
    # 转人工：把当前会话内容写成工单，后续可由人工客服处理。
    return {"ticket": mock_db.create_ticket(user_id, session_id, message, reason)}


def search_knowledge(query: str, top_k: int = 3) -> dict:
    # 调用本地向量知识库，返回最相关的知识条目。
    return {"hits": vector_store.search(query, top_k)}


# 工具注册表：Agent 通过名字查到真正要执行的函数。
TOOL_REGISTRY: dict[str, ToolFunc] = {
    "query_product": query_product,
    "recommend_product": recommend_product,
    "query_order": query_order,
    "list_coupon": list_coupon,
    "refund_order": refund_order,
    "transfer_human": transfer_human,
    "search_knowledge": search_knowledge,
}
