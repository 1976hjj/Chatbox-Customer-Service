from datetime import datetime


PRODUCTS = [
    {"sku": "P001", "name": "智利车厘子 500g", "price": 59.9, "stock": 128, "tags": ["水果", "当季"]},
    {"sku": "P002", "name": "新疆阿克苏苹果 2kg", "price": 29.9, "stock": 320, "tags": ["水果", "热销"]},
    {"sku": "P003", "name": "墨西哥牛油果 4粒", "price": 36.8, "stock": 46, "tags": ["水果", "轻食"]},
    {"sku": "P004", "name": "可生食鲜鸡蛋 20枚", "price": 25.9, "stock": 210, "tags": ["蛋品"]},
    {"sku": "P005", "name": "挪威三文鱼切片 200g", "price": 49.9, "stock": 34, "tags": ["水产", "冷链"]},
]

ORDERS = {
    "O20260617001": {
        "order_id": "O20260617001",
        "user_id": "u1001",
        "status": "配送中",
        "items": [{"sku": "P002", "name": "新疆阿克苏苹果 2kg", "quantity": 1}],
        "amount": 29.9,
        "delivery_eta": "2026-06-17 18:30",
        "logistics": [
            {"time": "2026-06-17 16:00", "status": "订单已拣货"},
            {"time": "2026-06-17 17:05", "status": "骑手已取货"},
        ],
    },
    "O20260616002": {
        "order_id": "O20260616002",
        "user_id": "u1001",
        "status": "已完成",
        "items": [{"sku": "P003", "name": "墨西哥牛油果 4粒", "quantity": 1}],
        "amount": 36.8,
        "delivery_eta": "2026-06-16 20:10",
        "logistics": [
            {"time": "2026-06-16 19:20", "status": "骑手已取货"},
            {"time": "2026-06-16 20:02", "status": "订单已签收"},
        ],
    },
}

COUPONS = {
    "u1001": [
        {"coupon_id": "C001", "name": "满99减15", "valid_until": "2026-06-30", "scope": "全品类"},
        {"coupon_id": "C002", "name": "水果满49减8", "valid_until": "2026-06-23", "scope": "水果"},
    ]
}

TICKETS: list[dict] = []
CHAT_HISTORY: list[dict] = []


def search_products(keyword: str | None = None) -> list[dict]:
    if not keyword:
        return PRODUCTS
    return [item for item in PRODUCTS if keyword in item["name"] or keyword in "".join(item["tags"])]


def get_order(order_id: str | None, user_id: str) -> dict | None:
    if order_id:
        order = ORDERS.get(order_id)
        if order and order["user_id"] == user_id:
            return order
        return order
    for order in ORDERS.values():
        if order["user_id"] == user_id:
            return order
    return None


def list_coupons(user_id: str) -> list[dict]:
    return COUPONS.get(user_id, [])


def create_ticket(user_id: str, session_id: str, message: str, reason: str) -> dict:
    ticket = {
        "ticket_id": f"T{len(TICKETS) + 1:06d}",
        "user_id": user_id,
        "session_id": session_id,
        "message": message,
        "reason": reason,
        "status": "待人工处理",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    TICKETS.append(ticket)
    return ticket


def save_chat(record: dict) -> None:
    CHAT_HISTORY.append(record)
