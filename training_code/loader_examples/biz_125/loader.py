def load(raw):
    # Read the raw data from the CSV file
    df = raw.read_table("ban_hang.csv")
    
    # Normalize column names
    df.columns = [
        "unit", "product_name", "customer_name", "qty_base", "line_amount",
        "status_raw", "qty", "order_date", "order_id", "unit_price"
    ]
    
    # Parse columns
    df["qty_base"] = parse_number(df["qty_base"])
    df["line_amount"] = parse_money(df["line_amount"])
    df["qty"] = parse_number(df["qty"])
    df["unit_price"] = parse_money(df["unit_price"])
    df["order_date"] = parse_date(df["order_date"])
    
    # Map order status
    status_map = {
        "Đã hoàn thành": "completed",
        "Huỷ đơn": "cancelled",
        "Đã trả hàng": "returned",
        "Đang xử lý": "delivering"
    }
    df["status"] = df["status_raw"].map(status_map)
    
    # Handle walk-in customers
    df["customer_id"] = df["customer_name"].where(df["customer_name"] != "Khách lẻ", None)
    df["customer_name"] = df["customer_name"].where(df["customer_name"] != "Khách lẻ", None)
    
    # Create orders table
    orders = (
        df.groupby("order_id")
        .agg(
            order_datetime=("order_date", "min"),
            order_date=("order_date", "min"),
            status=("status", "first"),
            status_raw=("status_raw", "first"),
            customer_id=("customer_id", "first"),
            customer_name=("customer_name", "first"),
            goods_total=("line_amount", "sum"),
        )
        .reset_index()
    )
    
    # Create order_lines table
    order_lines = df.copy()
    order_lines["line_no"] = order_lines.groupby("order_id").cumcount() + 1
    order_lines["is_fee"] = order_lines["product_name"].str.contains("Phí phục vụ")
    order_lines = order_lines[[
        "order_id", "line_no", "product_name", "unit", "qty", "qty_base",
        "unit_price", "line_amount", "is_fee"
    ]]
    
    # Create unit_conversions table
    conversion_data = [
        {"product_id": "Bia Tiger lon", "unit": "thùng", "factor": 24},
        {"product_id": "Bia Sài Gòn Special lon", "unit": "thùng", "factor": 24},
        {"product_id": "Coca-Cola lon", "unit": "thùng", "factor": 24},
        {"product_id": "Bia Tiger Bạc lon", "unit": "thùng", "factor": 24},
        {"product_id": "Nước suối", "unit": "thùng", "factor": 24},
        {"product_id": "Bia Heineken lon", "unit": "thùng", "factor": 24},
        {"product_id": "Bia Sài Gòn Lager chai", "unit": "két", "factor": 20},
        {"product_id": "7Up lon", "unit": "thùng", "factor": 24},
    ]
    unit_conversions = pd.DataFrame(conversion_data)
    
    # Create products table
    products = (
        df.groupby("product_name")
        .agg(base_unit=("unit", "first"))
        .reset_index()
    )
    products["product_id"] = products["product_name"]
    products.rename(columns={"product_name": "name"}, inplace=True)
    products = products[["product_id", "name", "base_unit"]]
    
    # Prepare metadata
    meta = {
        "industry": "nhà hàng",
        "has_customer_codes": False,
        "line_no_from_file": False,
        "data_end": "2026-05-11",
        "notes": [
            "Phí phục vụ là dòng hàng đặc biệt, không phải mặt hàng.",
            "Quy đổi đơn vị theo README."
        ],
        "extra_desc": {}
    }
    
    # Return tables and metadata
    tables = {
        "orders": orders,
        "order_lines": order_lines,
        "products": products,
        "unit_conversions": unit_conversions
    }
    return tables, meta
