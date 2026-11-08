def load(raw):
    import pandas as pd
    import numpy as np

    meta = {
        "industry": "nhà thuốc",
        "has_customer_codes": False,
        "line_no_from_file": False,
        "order_codes_from_file": True,
        "data_end": "2026-09-20",
        "notes": [
            "unit conversions provided in DanhSachQuyDoi",
            "batch-level tracking for sales and returns",
            "prescription drugs and active ingredients recorded",
        ],
        "extra_desc": {}
    }

    # Load `DonHang_ChiTiet.xlsx`
    orders_raw = raw.read_table("DonHang_ChiTiet.xlsx", sheet="Sheet1")
    orders_raw.columns = [
        "order_id", "order_date", "status_raw", "customer_name", "product_name",
        "qty", "unit_price", "line_amount", "unit"
    ]

    # Parse dates and money
    orders_raw["order_date"] = parse_datetime(orders_raw["order_date"])
    orders_raw["unit_price"] = parse_money(orders_raw["unit_price"])
    orders_raw["line_amount"] = parse_money(orders_raw["line_amount"])

    # Forward-fill order-level data
    orders_raw = ffill_group(orders_raw, key="order_id", cols=[
        "order_date", "status_raw", "customer_name"
    ])

    # Map order statuses
    status_mapping = {
        "Đã hoàn thành": "completed",
        "Đã hủy": "cancelled",
        "Đã hoàn hàng": "returned",
        "Đang giao dịch": "delivering"
    }
    orders_raw["status"] = map_status(orders_raw["status_raw"], status_mapping)

    # Load `DanhSachQuyDoi` for unit conversions
    conversions = raw.read_table("DuLieuXuat.xlsx", sheet="DanhSachQuyDoi")
    conversions.columns = ["product_name", "unit", "factor", "base_unit"]

    # Load `DanhSachSanPham` for product catalogue
    products_raw = raw.read_table("DuLieuXuat.xlsx", sheet="DanhSachSanPham")
    products_raw = products_raw.rename(columns={
        "Mã SKU": "product_id",
        "Tên sản phẩm": "name",
        "Đơn vị": "base_unit",
        "Loại sản phẩm": "category"
    })

    # Load `DanhSachThuoc` for prescription drugs
    drugs_raw = raw.read_table("DuLieuXuat.xlsx", sheet="DanhSachThuoc")
    drugs_raw = drugs_raw.rename(columns={
        "Mã SKU": "product_id",
        "Hoạt chất": "active_ingredient",
        "Thuốc kê đơn": "is_prescription"
    })

    # Merge product catalogue with prescription drugs
    products = pd.merge(products_raw, drugs_raw, on="product_id", how="left")
    products["is_prescription"] = products["is_prescription"].fillna(False)

    # Load `DonNhapHang` and `DonNhapHang_ChiTiet` for purchases
    purchases_header = raw.read_table("DuLieuXuat.xlsx", sheet="DonNhapHang")
    purchases_detail = raw.read_table("DuLieuXuat.xlsx", sheet="DonNhapHang_ChiTiet")
    purchases = pd.merge(purchases_detail, purchases_header, on="Mã đơn nhập")
    purchases = purchases.rename(columns={
        "Mã đơn nhập": "receipt_id",
        "Ngày nhập": "receipt_date",
        "Mã nhà cung cấp": "supplier",
        "Mã SKU": "product_id",
        "Mã lô": "batch_id",
        "Hạn sử dụng": "expiry_date",
        "Số lượng nhập": "qty_base"
    })
    purchases["amount"] = purchases["qty_base"] * purchases.get("unit_cost", np.nan)

    # Load `DonHang_Lo` for batch sales
    batch_sales_raw = raw.read_table("DuLieuXuat.xlsx", sheet="DonHang_Lo")
    batch_sales_raw = batch_sales_raw.rename(columns={
        "Mã đơn hàng": "order_id",
        "Mã lô": "batch_id",
        "Số lượng": "qty_base",
        "Số lượng trả": "qty_returned_base"
    })

    # Prepare `orders` table
    orders = orders_raw.groupby("order_id", as_index=False).agg({
        "order_date": "first",
        "status": "first",
        "status_raw": "first",
        "customer_name": "first",
        "line_amount": "sum"
    }).rename(columns={
        "order_date": "order_datetime",
        "line_amount": "goods_total"
    })
    orders["order_date"] = orders["order_datetime"].dt.date
    orders["customer_id"] = np.where(
        orders["customer_name"].str.lower().str.contains("khách lẻ|lẻ", na=False),
        None, None
    )
    orders["order_discount"] = 0.0
    orders["shipping_fee"] = 0.0
    orders["vat"] = 0.0
    orders["total_payable"] = orders["goods_total"]
    orders["paid_amount"] = np.nan
    orders["debt_remaining"] = np.nan
    orders["note"] = None

    # Prepare `order_lines` table
    order_lines = orders_raw.rename(columns={
        "order_id": "order_id",
        "product_name": "product_name",
        "qty": "qty",
        "unit": "unit",
        "unit_price": "unit_price",
        "line_amount": "line_amount"
    })
    order_lines = pd.merge(order_lines, conversions, on=["product_name", "unit"], how="left")
    order_lines["factor"] = order_lines["factor"].fillna(1)  # Default factor to 1 if no conversion is found
    order_lines["qty_base"] = order_lines["qty"] * order_lines["factor"]
    order_lines["line_no"] = order_lines.groupby("order_id").cumcount() + 1
    order_lines["product_id"] = None  # No product_id in sales data
    order_lines["cost_per_unit"] = np.nan
    order_lines["is_fee"] = False

    # Prepare `unit_conversions` table
    unit_conversions = conversions.rename(columns={
        "product_name": "product_id"
    })

    # Prepare `batch_sales` table
    batch_sales = batch_sales_raw.copy()

    # Return tables and meta
    tables = {
        "orders": orders,
        "order_lines": order_lines,
        "products": products,
        "unit_conversions": unit_conversions,
        "purchases": purchases,
        "batch_sales": batch_sales
    }
    return tables, meta
