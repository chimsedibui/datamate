def load(raw):
    # Read the raw data from the CSV file
    df = raw.read_table("ban_hang.csv")
    
    # Normalize column names
    df.columns = [
        "unit_price", "unit", "order_id", "qty_base", "qty", "product_name", 
        "line_amount", "customer_name", "order_date", "status_raw"
    ]
    
    # Parse and clean data
    df["unit_price"] = parse_money(df["unit_price"])
    df["qty_base"] = parse_number(df["qty_base"])
    df["qty"] = parse_qty_unit(df["qty"])["qty"]
    df["line_amount"] = parse_money(df["line_amount"])
    df["order_date"] = parse_date(df["order_date"])
    df["order_datetime"] = df["order_date"]  # No time information available, default to 00:00
    df["status_raw"] = df["status_raw"].str.strip()
    
    # Map status to canonical values
    status_map = {
        "đã lấy": "completed",
        "Đã hủy đơn": "cancelled",
        "Trả hàng": "returned",
        "Đang giao": "delivering"
    }
    df["status"] = df["status_raw"].map(status_map)
    
    # Handle walk-in customers
    df["customer_id"] = np.where(df["customer_name"].str.lower().str.contains("khách lẻ"), None, None)
    df["customer_name"] = np.where(df["customer_name"].str.lower().str.contains("khách lẻ"), None, df["customer_name"])
    
    # Create orders table
    orders = (
        df.groupby("order_id")
        .agg(
            order_datetime=("order_date", "first"),
            order_date=("order_date", "first"),
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
    order_lines["is_fee"] = False  # No indication of fees in the data
    order_lines = order_lines[[
        "order_id", "line_no", "product_name", "unit", "qty", "qty_base", 
        "unit_price", "line_amount", "is_fee"
    ]]
    
    # Create unit_conversions table
    conversion_data = [
        ("Xi măng Hà Tiên PCB40 bao 50kg", "bao", 20),
        ("Xi măng INSEE Power-S bao 50kg", "bao", 20),
        ("Cát xây tô", "m3", 4),
        ("Cát bê tông", "m3", 4),
        ("Cát san lấp", "m3", 4),
        ("Đá 1x2", "m3", 4),
        ("Đá 4x6", "m3", 4),
        ("Đá mi bụi", "m3", 4),
        ("Gạch ống 8x8x18 Tuynel", "viên", 1000),
        ("Gạch thẻ đặc 4x8x18", "viên", 1000),
        ("Băng keo điện Nano 20 yard", "cuộn", 10),
        ("Lưới B40 mạ kẽm khổ 1.5m", "m", 10),
        ("Ngói Đồng Nai 22 viên/m2", "viên", 1000),
    ]
    unit_conversions = pd.DataFrame(conversion_data, columns=["product_id", "unit", "factor"])
    
    # Create products table
    products = (
        df[["product_name"]]
        .drop_duplicates()
        .rename(columns={"product_name": "name"})
    )
    products["product_id"] = products["name"]
    products["base_unit"] = products["name"].map(
        {row[0]: row[1] for row in conversion_data}
    )
    products["active"] = True
    products = products[[
        "product_id", "name", "base_unit", "active"
    ]]
    
    # Prepare metadata
    meta = {
        "industry": "vật liệu xây dựng",
        "has_customer_codes": False,
        "line_no_from_file": False,
        "data_end": "2026-06-21",
        "notes": [
            "Units are recorded in the file and converted to base units using the provided conversion table.",
            "Customer codes are not recorded; customers are identified by name.",
        ],
        "extra_desc": {},
    }
    
    # Return tables and metadata
    tables = {
        "orders": orders,
        "order_lines": order_lines,
        "products": products,
        "unit_conversions": unit_conversions,
    }
    return tables, meta
