def load(raw):
    # Read all files
    don_hang = raw.read_table("don_hang.csv")
    chi_tiet_don = raw.read_table("chi_tiet_don.csv")
    san_pham = raw.read_table("san_pham.csv")
    khach_hang = raw.read_table("khach_hang.csv")
    thanh_toan = raw.read_table("thanh_toan.csv")

    # Parse orders table
    orders = don_hang.rename(columns={
        "Mã chứng từ": "order_id",
        "NgayBan": "order_datetime",
        "Xử lý": "status_raw",
        "Số KH": "customer_id",
        "Bán qua": "channel",
        "Phương thức TT": "payment_method_raw",
        "Tổng bill": "total_payable",
        "Khách trả": "paid_amount",
        "Còn lại": "debt_remaining"
    })
    orders["order_datetime"] = parse_datetime(orders["order_datetime"], default_year=2025)
    orders["order_date"] = orders["order_datetime"].dt.date
    orders["status"] = orders["status_raw"].str.lower().map({
        "completed": "completed",
        "cancelled": "cancelled",
        "returned": "returned",
        "shipping": "delivering"
    })
    orders["customer_name"] = orders["customer_id"].map(
        khach_hang.set_index("ma_kh")["Khách lẻ/sỉ"]
    )
    orders["customer_group"] = orders["customer_id"].map(
        khach_hang.set_index("ma_kh")["Nhóm"]
    )
    orders["payment_method"] = orders["payment_method_raw"].str.lower().map({
        "cash": "cash",
        "transfer": "transfer",
        "ewallet": "ewallet",
        "cod": "cod",
        "debt": "debt"
    })
    orders["order_discount"] = 0.0  # No explicit order discount column
    orders["shipping_fee"] = 0.0  # No explicit shipping fee column
    orders["vat"] = 0.0  # No explicit VAT column
    orders["note"] = None  # No explicit note column
    orders = orders[[
        "order_id", "order_datetime", "order_date", "status", "status_raw",
        "customer_id", "customer_name", "customer_group", "channel",
        "payment_method", "payment_method_raw", "total_payable", "paid_amount",
        "debt_remaining", "order_discount", "shipping_fee", "vat", "note"
    ]]

    # Parse order_lines table
    order_lines = chi_tiet_don.rename(columns={
        "Ma don": "order_id",
        "Thứ tự": "line_no",
        "ID SP": "product_id",
        "T.Tiền": "line_amount",
        "SL": "qty",
        "SL quy đổi": "qty_base",
        "Đơn vị (tính)": "unit",
        "Giá đơn vị": "unit_price"
    })
    order_lines["qty"] = parse_qty_unit(order_lines["qty"])["qty"]
    order_lines["unit_price"] = parse_money(order_lines["unit_price"])
    order_lines["line_amount"] = parse_money(order_lines["line_amount"])
    order_lines["line_discount"] = order_lines["qty"] * order_lines["unit_price"] - order_lines["line_amount"]
    order_lines["is_fee"] = order_lines["product_id"].str.contains("phí phục vụ", case=False, na=False)
    order_lines["product_name"] = order_lines["product_id"].map(
        san_pham.set_index("SKU")["Sản phẩm"]
    )
    order_lines = order_lines[[
        "order_id", "line_no", "product_id", "product_name", "unit", "qty", "qty_base",
        "unit_price", "line_discount", "line_amount", "is_fee"
    ]]

    # Parse products table
    products = san_pham.rename(columns={
        "SKU": "product_id",
        "Sản phẩm": "name",
        "Nhom hang": "category",
        "Giá": "sell_price",
        "Đơn vị": "base_unit"
    })
    products["sell_price"] = parse_money(products["sell_price"])
    products["cost_price"] = None  # No explicit cost price column
    products["is_prescription"] = False  # Not a pharmacy
    products["active_ingredient"] = None  # Not a pharmacy
    products["size"] = None  # No explicit size column
    products["is_prepared"] = False  # No explicit prepared column
    products["active"] = True  # Assume all products are active
    products["aliases"] = None  # No explicit aliases column
    products["brand"] = None  # No explicit brand column
    products = products[[
        "product_id", "name", "aliases", "category", "brand", "base_unit",
        "sell_price", "cost_price", "is_prescription", "active_ingredient",
        "size", "is_prepared", "active"
    ]]

    # Parse customers table
    customers = khach_hang.rename(columns={
        "ma_kh": "customer_id",
        "Khách lẻ/sỉ": "name",
        "Nhóm": "group",
        "Phone": "phone"
    })
    customers["registered_date"] = None  # No explicit registration date column
    customers = customers[[
        "customer_id", "name", "group", "phone", "registered_date"
    ]]

    # Parse payments table
    payments = thanh_toan.rename(columns={
        "Số HĐ": "order_id",
        "Thời gian thu": "paid_datetime",
        "Tiền": "amount",
        "Phương thức": "method",
        "Mã phiếu thu": "payment_id"
    })
    payments["paid_datetime"] = parse_datetime(payments["paid_datetime"])
    payments["amount"] = parse_money(payments["amount"])
    payments["method"] = payments["method"].str.lower().map({
        "cash": "cash",
        "transfer": "transfer",
        "ewallet": "ewallet",
        "cod": "cod"
    })
    payments["kind"] = "payment"  # Default kind
    payments = payments[[
        "payment_id", "order_id", "paid_datetime", "amount", "method", "kind"
    ]]

    # Unit conversions table
    unit_conversions = pd.DataFrame([
        {"product_id": "SP0046", "unit": "hộp", "factor": 6},
        {"product_id": "SP0046", "unit": "cái", "factor": 1},
        {"product_id": "SP0047", "unit": "hộp", "factor": 6},
        {"product_id": "SP0047", "unit": "cái", "factor": 1}
    ])

    # Meta information
    meta = {
        "industry": "cà phê",
        "has_customer_codes": True,
        "line_no_from_file": True,
        "data_end": "2026-08-19",
        "notes": [
            "Unit conversions for Bánh croissant and Bánh tiramisu provided in README",
            "Debt is recorded in the 'Còn lại' column in don_hang.csv"
        ],
        "extra_desc": {},
        "dropped_duplicate_lines": 0
    }

    # Return tables and meta
    tables = {
        "orders": orders,
        "order_lines": order_lines,
        "products": products,
        "customers": customers,
        "payments": payments,
        "unit_conversions": unit_conversions
    }
    return tables, meta
