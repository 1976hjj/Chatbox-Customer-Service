from typing import Any, Callable

from app.repositories import mock_db
from app.services.vector_store import vector_store


ToolFunc = Callable[..., Any]


def query_product(product: str | None = None) -> dict:
    products = mock_db.search_products(product)
    return {"products": products[:5]}


def recommend_product(product: str | None = None) -> dict:
    products = mock_db.search_products(product) or mock_db.PRODUCTS
    return {"recommendations": sorted(products, key=lambda item: item["stock"], reverse=True)[:3]}


def query_order(order_id: str | None, user_id: str) -> dict:
    order = mock_db.get_order(order_id, user_id)
    return {"order": order}


def list_coupon(user_id: str) -> dict:
    return {"coupons": mock_db.list_coupons(user_id)}


def refund_order(order_id: str | None, user_id: str, reason: str) -> dict:
    order = mock_db.get_order(order_id, user_id)
    if not order:
        return {"accepted": False, "message": "未找到可退款订单"}
    ticket = mock_db.create_ticket(user_id, "refund", reason, "refund")
    return {"accepted": True, "order": order, "ticket": ticket}


def transfer_human(user_id: str, session_id: str, message: str, reason: str) -> dict:
    return {"ticket": mock_db.create_ticket(user_id, session_id, message, reason)}


def search_knowledge(query: str, top_k: int = 3) -> dict:
    return {"hits": vector_store.search(query, top_k)}


TOOL_REGISTRY: dict[str, ToolFunc] = {
    "query_product": query_product,
    "recommend_product": recommend_product,
    "query_order": query_order,
    "list_coupon": list_coupon,
    "refund_order": refund_order,
    "transfer_human": transfer_human,
    "search_knowledge": search_knowledge,
}
