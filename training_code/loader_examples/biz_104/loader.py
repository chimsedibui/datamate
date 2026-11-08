def load(raw):
    import pandas as pd
    import numpy as np

    meta = {
        "industry": "cà phê",
        "has_customer_codes": False,
        "line_no_from_file": False,
        "data_end": "2026-06-02",
        "notes": [
            "Không có mã khách; 'Khách lẻ' / khách vãng lai để null.",
            "Số lượng quy đổi theo DanhSachQuyDoi.xlsx; qty_base = qty × tỉ lệ quy đổi khi dòng dùng đơn vị lớn.",
            "Trạng thái 'Đang giao dịch' được map sang delivering.",
            "Dữ liệu có đến hết 2026-06-02."
        ],
        "extra_desc": {},
    }

    def _first_sheet_df(filename):
        sheets = raw.sheet_names(filename)
        sheet = sheets[0] if sheets else None
        return raw.read_table(filename, sheet=sheet)

    # --- Read raw tables ---
    sales_file = next(f for f in raw.list_files() if "DonHang_ChiTiet" in f)
    conv_file = next(f for f in raw.list_files() if "DanhSachQuyDoi" in f)

    sales = _first_sheet_df(sales_file)
    conv = _first_sheet_df(conv_file)

    # --- Standardize column names ---
    sales = sales.rename(columns={
        "Mã đơn hàng": "order_id",
        "Ngày đặt hàng": "order_date_raw",
        "Trạng thái đơn hàng": "status_raw",
        "Tên khách hàng": "customer_name_raw",
        "Tên sản phẩm": "product_name",
        "Số lượng sản phẩm": "qty_raw",
        "Giá sản phẩm": "unit_price_raw",
        "Tổng tiền": "line_amount_raw",
        "Đơn vị tính": "unit_raw",
    })

    conv = conv.rename(columns={
        "Tên sản phẩm": "product_name",
        "Đơn vị quy đổi": "conv_unit_raw",
        "Tỉ lệ quy đổi": "factor_raw",
        "Đơn vị cơ bản": "base_unit",
    })

    # --- Clean sales core columns ---
    for c in ["order_id", "order_date_raw", "status_raw", "customer_name_raw", "product_name", "unit_raw"]:
        if c in sales.columns:
            sales[c] = sales[c].where(sales[c].notna(), np.nan)
            sales[c] = sales[c].replace("", np.nan)

    # Forward-fill order metadata across missing cells in the row stream
    sales["order_id"] = sales["order_id"].ffill()

    # Parse dates / numerics
    sales["order_datetime"] = parse_datetime(sales["order_date_raw"])
    sales["qty"] = parse_number(sales["qty_raw"])
    sales["unit_price"] = parse_money(sales["unit_price_raw"])
    sales["line_amount"] = parse_money(sales["line_amount_raw"])
    sales["unit"] = sales["unit_raw"]

    # Walk-in customers -> null
    cust = sales["customer_name_raw"]
    cust_norm = cust.map(lambda x: norm_text(x) if pd.notna(x) else None)
    walkin_mask = cust_norm.isin(["khách lẻ", "khách vãng lai"]) | cust.isna()
    sales["customer_name"] = cust.where(~walkin_mask, None)

    # Status mapping
    status_map = {
        "Đã hoàn thành": "completed",
        "Đã hủy": "cancelled",
        "Đã hoàn hàng": "returned",
        "Đang giao dịch": "delivering",
    }
    sales["status"] = map_status(sales["status_raw"], status_map)

    # --- Read conversion table ---
    conv["product_key"] = conv["product_name"].map(lambda x: norm_text(x) if pd.notna(x) else None)
    conv["conv_unit_key"] = conv["conv_unit_raw"].map(lambda x: norm_text(x) if pd.notna(x) else None)
    conv["factor"] = parse_number(conv["factor_raw"])

    conv_lookup = conv[["product_key", "conv_unit_key", "factor", "base_unit"]].drop_duplicates()

    sales["product_key"] = sales["product_name"].map(lambda x: norm_text(x) if pd.notna(x) else None)
    sales["unit_key"] = sales["unit"].map(lambda x: norm_text(x) if pd.notna(x) else None)

    sales = sales.merge(conv_lookup, on="product_key", how="left", suffixes=("", "_conv"))
    sales["qty_base"] = np.where(
        sales["conv_unit_key"].notna() & (sales["unit_key"] == sales["conv_unit_key"]),
        sales["factor"],
        1.0,
    )
    sales["qty_base"] = sales["qty"] * sales["qty_base"]

    # Keep only valid line rows for order_lines
    line_df = sales.loc[sales["product_name"].notna() & sales["qty"].notna()].copy()
    line_df = line_df.reset_index(drop=True)
    line_df["line_no"] = line_df.groupby("order_id", sort=False).cumcount() + 1
    line_df["product_id"] = line_df["product_name"]
    line_df["is_fee"] = False

    order_lines = line_df[[
        "order_id",
        "line_no",
        "product_id",
        "product_name",
        "unit",
        "qty",
        "qty_base",
        "unit_price",
        "line_amount",
        "is_fee",
    ]].copy()

    # --- Orders (one row per order_id) ---
    grp = sales.groupby("order_id", sort=False, dropna=False)

    orders = pd.DataFrame({
        "order_id": grp["order_id"].first(),
        "order_datetime": grp["order_datetime"].first(),
        "status_raw": grp["status_raw"].first(),
        "customer_name": grp["customer_name"].first(),
    }).reset_index(drop=True)

    orders["status"] = map_status(orders["status_raw"], status_map)
    orders["order_date"] = orders["order_datetime"].dt.date

    # Keep only core/observed columns
    orders = orders[["order_id", "order_datetime", "order_date", "status", "status_raw", "customer_name"]]

    # --- Non-core tables inside try/except ---
    tables = {
        "orders": orders,
        "order_lines": order_lines,
    }

    try:
        # Products: union of sold products and conversion-table products
        sold_products = pd.Index(sales["product_name"].dropna().astype(str).unique())
        conv_products = pd.Index(conv["product_name"].dropna().astype(str).unique())
        all_products = pd.Index(sorted(set(sold_products.tolist()) | set(conv_products.tolist())))

        # Base unit: from conversion table when available, else most common observed sales unit
        unit_mode = (
            sales.dropna(subset=["product_name", "unit"])
                 .groupby("product_name")["unit"]
                 .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
        )
        conv_base_map = (
            conv.dropna(subset=["product_name", "base_unit"])
                .drop_duplicates(subset=["product_name"])
                .set_index("product_name")["base_unit"]
                .to_dict()
        )

        products = pd.DataFrame({"product_id": all_products})
        products["name"] = products["product_id"]
        products["base_unit"] = products["product_id"].map(conv_base_map).fillna(
            products["product_id"].map(unit_mode)
        )

        # Keep minimal canonical columns only
        products = products[["product_id", "name", "base_unit"]].drop_duplicates()

        tables["products"] = products

        unit_conversions = conv.dropna(subset=["product_name", "conv_unit_raw", "factor"]).copy()
        unit_conversions["product_id"] = unit_conversions["product_name"]
        unit_conversions["unit"] = unit_conversions["conv_unit_raw"]
        unit_conversions["factor"] = unit_conversions["factor"]

        unit_conversions = unit_conversions[["product_id", "unit", "factor"]].drop_duplicates()

        tables["unit_conversions"] = unit_conversions
    except Exception as e:
        meta.setdefault("table_errors", {})["products_or_unit_conversions"] = repr(e)

    return tables, meta
