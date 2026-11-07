"""Canonical semantic schema shared by every business.

Every business's raw files are mapped (by LLM-generated loader code) into these
tables. Metric functions only ever read canonical tables, so each rulebook rule
is implemented once.

Conventions:
    * money is VND as float (rounded only at answer time)
    * qty_base = quantity converted to the product's base unit (đơn vị cơ bản)
    * dates are pandas Timestamps; *_date columns are normalised to midnight
    * status uses the enums below; anything unknown -> 'completed' only if the
      README says so, otherwise keep the raw value in status_raw
    * a column that the business does not record is simply absent (never invented)
"""
from __future__ import annotations

from dataclasses import dataclass, field

ORDER_STATUS = ("completed", "cancelled", "returned", "delivering")
PAYMENT_METHOD = ("cash", "transfer", "ewallet", "card", "cod", "debt", "installment", "other")


@dataclass
class Col:
    name: str
    dtype: str  # str | float | int | datetime | date | bool
    desc: str
    required: bool = False


@dataclass
class Table:
    name: str
    desc: str
    cols: list[Col]
    key: list[str] = field(default_factory=list)

    def col(self, name: str) -> Col | None:
        return next((c for c in self.cols if c.name == name), None)


TABLES: dict[str, Table] = {}


def _t(name, desc, cols, key=()):
    TABLES[name] = Table(name, desc, [Col(*c) for c in cols], list(key))


_t("orders", "One row per order/invoice (đơn hàng, hoá đơn, phiếu).", [
    ("order_id", "str", "order code; synthesise a stable id if the file has none", True),
    ("order_datetime", "datetime", "order creation date+time (ngày giờ đặt/tạo đơn); time may be 00:00 if unknown", True),
    ("order_date", "date", "order_datetime normalised to the day", True),
    ("status", "str", f"one of {ORDER_STATUS}", True),
    ("status_raw", "str", "original status text"),
    ("customer_id", "str", "customer code; null for walk-in (khách lẻ / vãng lai)"),
    ("customer_name", "str", "customer name as written on the order; null for walk-in"),
    ("customer_group", "str", "customer group/segment column if recorded on the order"),
    ("employee", "str", "seller / staff / technician name or code"),
    ("branch", "str", "branch / store"),
    ("channel", "str", "sales channel (tại quầy, online, sàn TMĐT, zalo...)"),
    ("payment_method", "str", f"payment method recorded ON THE ORDER, one of {PAYMENT_METHOD}"),
    ("payment_method_raw", "str", "original payment method text"),
    ("goods_total", "float", "sum of line amounts as recorded on the order header (tổng tiền hàng)"),
    ("order_discount", "float", "order-level discount / voucher (positive number)"),
    ("shipping_fee", "float", "delivery fee charged to customer"),
    ("vat", "float", "VAT amount"),
    ("total_payable", "float", "amount the customer must pay for the whole order (after order discount, incl. shipping)"),
    ("paid_amount", "float", "amount paid as recorded on the order header"),
    ("debt_remaining", "float", "unpaid remainder as recorded on the order header"),
    ("note", "str", "free-text note"),
], key=["order_id"])

_t("order_lines", "One row per item line inside an order.", [
    ("order_id", "str", "FK orders.order_id", True),
    ("line_no", "int", "line number inside the order (1..n)"),
    ("product_id", "str", "product code; null if the shop has no codes"),
    ("product_name", "str", "product name as written on the line", True),
    ("unit", "str", "unit of the line (đơn vị tính of the sale)"),
    ("qty", "float", "quantity in the line's unit", True),
    ("qty_base", "float", "quantity converted to the product's base unit", True),
    ("unit_price", "float", "price per line unit BEFORE line discount"),
    ("line_discount", "float", "line-level discount amount (positive)"),
    ("line_amount", "float", "line total AFTER line discount (thành tiền); this is revenue", True),
    ("cost_per_unit", "float", "cost price per line unit recorded on the sale line (giá vốn lúc bán)"),
    ("is_fee", "bool", "True for non-merchandise lines such as 'Phí phục vụ' (still revenue, never ranked)"),
], key=["order_id", "line_no"])

_t("products", "Product catalogue (danh mục hàng).", [
    ("product_id", "str", "product code; use the name if no code exists", True),
    ("name", "str", "canonical product name", True),
    ("aliases", "str", "other spellings / short names separated by '|'"),
    ("category", "str", "product group (nhóm hàng)"),
    ("brand", "str", "brand / manufacturer"),
    ("base_unit", "str", "base unit (đơn vị cơ bản)"),
    ("sell_price", "float", "current list price per base unit"),
    ("cost_price", "float", "catalogue cost per base unit (giá vốn trong danh mục)"),
    ("is_prescription", "bool", "pharmacy: prescription drug"),
    ("active_ingredient", "str", "pharmacy: active ingredient(s) exactly as recorded"),
    ("size", "str", "size / stage column (e.g. milk 'Số 2', diaper 'L')"),
    ("is_prepared", "bool", "made on site (drinks, dishes) rather than bought for resale"),
    ("active", "bool", "still on sale"),
], key=["product_id"])

_t("unit_conversions", "Unit conversion factors to the base unit.", [
    ("product_id", "str", "FK products.product_id", True),
    ("unit", "str", "larger unit name", True),
    ("factor", "float", "how many base units in one `unit`", True),
], key=["product_id", "unit"])

_t("customers", "Customer master.", [
    ("customer_id", "str", "customer code", True),
    ("name", "str", "customer name"),
    ("group", "str", "customer group"),
    ("phone", "str", "phone"),
    ("registered_date", "date", "member registration date"),
], key=["customer_id"])

_t("payments", "Each money receipt against an order (phiếu thu, lần trả).", [
    ("payment_id", "str", "receipt code"),
    ("order_id", "str", "FK orders.order_id", True),
    ("paid_datetime", "datetime", "when the money was received", True),
    ("amount", "float", "amount received this time", True),
    ("method", "str", f"one of {PAYMENT_METHOD}"),
    ("kind", "str", "'payment' (default), 'installment', 'refund', 'deposit'"),
])

_t("purchases", "Purchase receipt lines (phiếu nhập hàng), one row per line.", [
    ("receipt_id", "str", "receipt code", True),
    ("receipt_date", "date", "receipt date (ngày phiếu)", True),
    ("supplier", "str", "supplier name"),
    ("status", "str", "'ok' or 'cancelled'"),
    ("product_id", "str", "FK products.product_id"),
    ("product_name", "str", "product name on the receipt", True),
    ("unit", "str", "purchase unit"),
    ("qty", "float", "quantity in purchase unit", True),
    ("qty_base", "float", "quantity in base unit"),
    ("unit_cost", "float", "purchase price per purchase unit"),
    ("amount", "float", "line amount", True),
    ("batch_id", "str", "batch / lot code"),
    ("expiry_date", "date", "expiry date of the batch"),
    ("note", "str", "e.g. 'nhập đầu kỳ'"),
])

_t("stock", "Inventory snapshot as given by the shop.", [
    ("product_id", "str", "FK products.product_id"),
    ("product_name", "str", "product name", True),
    ("qty_base", "float", "stock in base unit", True),
    ("as_of_date", "date", "snapshot date (usually last data day)"),
])

_t("stock_moves", "Stock ledger (thẻ kho): every in/out movement.", [
    ("product_id", "str", "FK products.product_id", True),
    ("moved_at", "datetime", "movement time", True),
    ("qty_base", "float", "signed quantity in base unit (+in, -out)", True),
    ("move_type", "str", "sale, purchase, return, adjust..."),
    ("doc_id", "str", "source document"),
    ("balance_after", "float", "stock after this movement"),
])

_t("batches", "Batches / lots (lô hàng), pharmacy etc.", [
    ("batch_id", "str", "batch code", True),
    ("product_id", "str", "FK products.product_id", True),
    ("expiry_date", "date", "expiry date", True),
    ("qty_in_base", "float", "quantity received in base unit"),
    ("unit_cost_base", "float", "purchase cost per base unit of this batch"),
    ("receipt_id", "str", "FK purchases.receipt_id"),
    ("received_date", "date", "date received"),
])

_t("batch_sales", "Which batch each sale line consumed.", [
    ("order_id", "str", "FK orders.order_id", True),
    ("line_no", "int", "FK order_lines.line_no"),
    ("product_id", "str", "product"),
    ("batch_id", "str", "FK batches.batch_id", True),
    ("qty_base", "float", "quantity taken from the batch (base unit)", True),
    ("qty_returned_base", "float", "quantity returned back to the batch"),
])

_t("price_history", "Selling price changes.", [
    ("product_id", "str", "FK products.product_id", True),
    ("effective_date", "date", "date the new price applies from", True),
    ("price_base", "float", "new price per base unit", True),
    ("old_price_base", "float", "previous price"),
])

_t("expenses", "Operating expenses (sổ chi phí).", [
    ("expense_date", "date", "date (or first day of month if only the month is known)", True),
    ("category", "str", "expense type"),
    ("amount", "float", "amount", True),
    ("note", "str", "description"),
])

_t("returns", "Return slips (phiếu trả hàng).", [
    ("return_id", "str", "return slip code"),
    ("order_id", "str", "FK orders.order_id", True),
    ("return_datetime", "datetime", "when returned"),
    ("refund_amount", "float", "money refunded"),
    ("reason", "str", "reason"),
])

_t("deliveries", "Delivery trips / shipments (đợt giao).", [
    ("delivery_id", "str", "delivery slip code"),
    ("order_id", "str", "FK orders.order_id", True),
    ("delivery_date", "date", "delivery date", True),
    ("status", "str", "delivery status"),
    ("note", "str", "note"),
])

CORE_TABLES = ("orders", "order_lines")


def schema_prompt(tables: list[str] | None = None) -> str:
    """Render the schema as compact text for LLM prompts."""
    out = []
    for t in TABLES.values():
        if tables and t.name not in tables:
            continue
        out.append(f"### {t.name} — {t.desc}")
        for c in t.cols:
            req = " (required)" if c.required else ""
            out.append(f"- {c.name}: {c.dtype}{req} — {c.desc}")
    return "\n".join(out)
