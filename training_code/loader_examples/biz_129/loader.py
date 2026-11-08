def load(raw):
    # Read all files
    orders = raw.read_table("don_hang.csv")
    order_lines = raw.read_table("chi_tiet_don.csv")
    products = raw.read_table("san_pham.csv")
    customers = raw.read_table("khach_hang.csv")
    payments = raw.read_table("thanh_toan.csv")

    # Map orders table
    orders["order_id"] = orders["Số phiếu"]
    orders["order_datetime"] = pd.to_datetime(orders["Ngày bán"])
    orders["order_date"] = orders["order_datetime"].dt.date
    orders["status_raw"] = orders["Tình trạng đơn"]
    orders["status"] = orders["Tình trạng đơn"].replace({
        "completed": "completed",
        "cancelled": "cancelled",
        "returned": "returned",
        "shipping": "delivering"
    })
    orders["customer_id"] = orders["Mã khách mua"]
    orders["customer_name"] = orders["Mã khách mua"].map(
        customers.set_index("Mã khách lẻ/sỉ")["KH"]
    )
    orders["customer_group"] = orders["Mã khách mua"].map(
        customers.set_index("Mã khách lẻ/sỉ")["Nhóm"]
    )
    orders["payment_method_raw"] = orders["Hình thức thanh toán"]
    orders["payment_method"] = orders["Hình thức thanh toán"].replace({
        "cash": "cash",
        "transfer": "transfer",
        "ewallet": "ewallet",
        "cod": "cod",
        "debt": "debt"
    })
    orders["goods_total"] = None  # Drop goods_total as it doesn't match line_amount
    orders["order_discount"] = None
    orders["shipping_fee"] = None
    orders["vat"] = None
    orders["total_payable"] = orders["Phải thu"].astype(float)
    orders["paid_amount"] = orders["Thanh toán"].astype(float)
    orders["debt_remaining"] = orders["Dư nợ"].astype(float)
    orders["note"] = None

    orders_table = orders[[
        "order_id", "order_datetime", "order_date", "status", "status_raw",
        "customer_id", "customer_name", "customer_group", "payment_method",
        "payment_method_raw", "order_discount", "shipping_fee", "vat",
        "total_payable", "paid_amount", "debt_remaining", "note"
    ]]

    # Map order_lines table
    order_lines["order_id"] = order_lines["Số CT"]
    order_lines["line_no"] = order_lines["TT dòng"].astype(int)
    order_lines["product_id"] = order_lines["Mã hàng bán"]
    order_lines["product_name"] = order_lines["Mã hàng bán"].map(
        products.set_index("Mã HH")["Tên sản phẩm"]
    )
    order_lines["unit"] = order_lines["ĐV tính"]
    order_lines["qty"] = order_lines["S.L"].astype(float)
    order_lines["qty_base"] = order_lines["Quy ra ĐV lẻ"].astype(float)
    order_lines["unit_price"] = order_lines["Giá 1 đv"].astype(float)
    order_lines["line_discount"] = order_lines["Giảm giá"].astype(float)
    order_lines["line_amount"] = order_lines["Tiền bán"].apply(parse_money)
    order_lines["cost_per_unit"] = None
    order_lines["is_fee"] = order_lines["product_name"].str.contains("phí phục vụ", case=False, na=False)

    order_lines_table = order_lines[[
        "order_id", "line_no", "product_id", "product_name", "unit", "qty",
        "qty_base", "unit_price", "line_discount", "line_amount", "cost_per_unit", "is_fee"
    ]]

    # Map products table
    products["product_id"] = products["Mã HH"]
    products["name"] = products["Tên sản phẩm"]
    products["aliases"] = None
    products["category"] = products["Nhom hang"]
    products["brand"] = None
    products["base_unit"] = products["Đơn vị tính"]
    products["sell_price"] = products["Giá hiện tại"].astype(float)
    products["cost_price"] = None
    products["is_prescription"] = None
    products["active_ingredient"] = None
    products["size"] = None
    products["is_prepared"] = None
    products["active"] = True

    products_table = products[[
        "product_id", "name", "aliases", "category", "brand", "base_unit",
        "sell_price", "cost_price", "is_prescription", "active_ingredient",
        "size", "is_prepared", "active"
    ]]

    # Map customers table
    customers["customer_id"] = customers["Mã khách lẻ/sỉ"]
    customers["name"] = customers["KH"]
    customers["group"] = customers["Nhóm"]
    customers["phone"] = customers["Số điện thoại"]
    customers["registered_date"] = None

    customers_table = customers[[
        "customer_id", "name", "group", "phone", "registered_date"
    ]]

    # Map payments table
    payments["payment_id"] = payments["Mã GD"]
    payments["order_id"] = payments["Mã giao dịch"]
    payments["paid_datetime"] = pd.to_datetime(payments["Thời điểm"])
    payments["amount"] = payments["Tiền"].astype(float)
    payments["method"] = payments["PTTT"].replace({
        "cash": "cash",
        "transfer": "transfer",
        "ewallet": "ewallet",
        "cod": "cod"
    })
    payments["kind"] = "payment"

    payments_table = payments[[
        "payment_id", "order_id", "paid_datetime", "amount", "method", "kind"
    ]]

    # Map unit_conversions table
    unit_conversions_data = [
        {"product_id": "SP0002", "unit": "thùng", "factor": 24},
        {"product_id": "SP0006", "unit": "thùng", "factor": 24},
        {"product_id": "SP0007", "unit": "thùng", "factor": 24},
        {"product_id": "SP0008", "unit": "thùng", "factor": 24},
        {"product_id": "SP0009", "unit": "thùng", "factor": 24},
        {"product_id": "SP0010", "unit": "thùng", "factor": 24},
        {"product_id": "SP0011", "unit": "két", "factor": 20},
        {"product_id": "SP0012", "unit": "thùng", "factor": 24},
    ]
    unit_conversions_table = pd.DataFrame(unit_conversions_data)

    # Meta information
    meta = {
        "industry": "tạp hoá",
        "has_customer_codes": True,
        "line_no_from_file": True,
        "data_end": "2026-08-14",
        "notes": [
            "Phí phục vụ là dòng hàng, không phải order-level adjustment",
            "Công nợ được ghi trong bảng thanh_toan.csv",
            "Quy đổi đơn vị theo README"
        ],
        "extra_desc": {},
        "dropped_duplicate_lines": 0
    }

    return {
        "orders": orders_table,
        "order_lines": order_lines_table,
        "products": products_table,
        "customers": customers_table,
        "payments": payments_table,
        "unit_conversions": unit_conversions_table
    }, meta
