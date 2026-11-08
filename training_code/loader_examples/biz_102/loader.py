def load(raw):
    import pandas as pd
    import numpy as np
    import re

    meta = {
        "industry": "tạp hoá",
        "has_customer_codes": False,
        "line_no_from_file": False,
        "data_end": "2026-06-16",
        "notes": [
            "DonHang_ChiTiet.xlsx là file Excel nhiều sheet theo tháng; mỗi dòng là một sản phẩm của đơn.",
            "Mã đơn hàng lặp lại theo từng dòng sản phẩm; line_no được sinh theo thứ tự dòng trong mỗi đơn.",
            "Trạng thái 'Đã hoàn thành' → completed, 'Đã hủy' → cancelled, 'Đã hoàn hàng' → returned.",
            "Tên khách dạng khách lẻ / vãng lai được coi là walk-in và để null; ngoài 'lẻ' còn có thể ghi tắt là 'kl'.",
            "Số lượng có thể kèm luôn đơn vị trong cùng ô (ví dụ '1 bao 25kg'); quy đổi đơn vị lấy từ DanhSachQuyDoi.xlsx.",
            "Không có mã khách; khách chỉ là tên ghi trên đơn nếu không phải khách lẻ.",
        ],
        "extra_desc": {},
    }

    # -------------------------
    # Helpers
    # -------------------------
    sales_file = next((f for f in raw.list_files() if f.lower().endswith("donhang_chitiet.xlsx")), None)
    conv_file = next((f for f in raw.list_files() if f.lower().endswith("danhsachquydoi.xlsx")), None)

    def _clean_str_series(s):
        return s.astype("string").str.strip()

    def _is_walkin(x):
        if pd.isna(x):
            return True
        t = fold_text(str(x))
        t2 = re.sub(r"\s+", "", t)
        return t in {
            "",
            "le",
            "khach le",
            "khach vang lai",
            "vang lai",
            "khach le ",
        } or t2 in {"kl", "khachle", "khachvanglai", "vanglai"}

    # -------------------------
    # Read source files
    # -------------------------
    sales_parts = []
    if sales_file is not None:
        for sh in raw.sheet_names(sales_file):
            df = raw.read_table(sales_file, sh)
            df = df.copy()
            df["__sheet"] = sh
            sales_parts.append(df)
    sales = pd.concat(sales_parts, ignore_index=True) if sales_parts else pd.DataFrame()

    conv = None
    if conv_file is not None:
        conv_parts = []
        for sh in raw.sheet_names(conv_file):
            cdf = raw.read_table(conv_file, sh)
            cdf = cdf.copy()
            cdf["__sheet"] = sh
            conv_parts.append(cdf)
        conv = pd.concat(conv_parts, ignore_index=True) if conv_parts else pd.DataFrame()

    # -------------------------
    # Unit conversions
    # -------------------------
    unit_map = {}       # (product_id, unit) -> factor
    base_unit_map = {}  # product_id -> base_unit

    if conv is not None and not conv.empty:
        c = conv.rename(columns={
            "Tên sản phẩm": "product_name",
            "Đơn vị quy đổi": "unit",
            "Tỉ lệ quy đổi": "factor",
            "Đơn vị cơ bản": "base_unit",
        }).copy()
        c["product_id"] = _clean_str_series(c["product_name"])
        c["unit"] = _clean_str_series(c["unit"])
        c["base_unit"] = _clean_str_series(c["base_unit"])
        c["factor"] = parse_number(c["factor"])

        for _, r in c.iterrows():
            pid = r.get("product_id")
            unit = r.get("unit")
            factor = r.get("factor")
            base_unit = r.get("base_unit")
            if pd.isna(pid) or pd.isna(unit) or pd.isna(factor):
                continue
            pid = str(pid)
            unit = str(unit)
            unit_map[(pid, unit)] = float(factor)
            if pd.notna(base_unit) and pid not in base_unit_map:
                base_unit_map[pid] = str(base_unit)

    # -------------------------
    # Sales: orders + order_lines
    # -------------------------
    orders = pd.DataFrame()
    order_lines = pd.DataFrame()

    if not sales.empty:
        s = sales.rename(columns={
            "Mã đơn hàng": "order_id",
            "Ngày đặt hàng": "order_date_raw",
            "Trạng thái đơn hàng": "status_raw",
            "Tên khách hàng": "customer_name_raw",
            "Tên sản phẩm": "product_name",
            "Số lượng sản phẩm": "qty_raw",
            "Giá sản phẩm": "unit_price_raw",
            "Tổng tiền": "line_amount_raw",
            "Đơn vị tính": "unit_raw",
        }).copy()

        s["order_id"] = _clean_str_series(s["order_id"])
        s["status_raw"] = _clean_str_series(s["status_raw"])
        s["customer_name_raw"] = _clean_str_series(s["customer_name_raw"])
        s["product_name"] = _clean_str_series(s["product_name"])
        s["unit_raw"] = _clean_str_series(s["unit_raw"])

        s["order_datetime"] = parse_datetime(s["order_date_raw"])
        s["order_date"] = s["order_datetime"].dt.date

        s["unit_price"] = parse_money(s["unit_price_raw"])
        s["line_amount"] = parse_money(s["line_amount_raw"])

        qty_unit = parse_qty_unit(s["qty_raw"])
        s["qty"] = qty_unit["qty"]
        s["unit"] = s["unit_raw"].where(s["unit_raw"].notna() & (s["unit_raw"] != ""), qty_unit["unit"])
        s["qty"] = s["qty"].where(s["qty"].notna(), parse_number(s["qty_raw"]))

        # Walk-in handling
        s["customer_name"] = s["customer_name_raw"].where(~s["customer_name_raw"].apply(_is_walkin), pd.NA)

        # Status mapping
        s["status"] = map_status(
            s["status_raw"],
            {
                "đã hoàn thành": "completed",
                "hoàn thành": "completed",
                "đã hủy": "cancelled",
                "hủy": "cancelled",
                "đã hoàn hàng": "returned",
                "hoàn hàng": "returned",
                "đã giao": "completed",
                "đang giao": "delivering",
                "chờ giao": "delivering",
            },
        )

        # Unit factor lookup
        def _factor_lookup(pid, unit):
            if pd.isna(pid) or pd.isna(unit):
                return np.nan
            pid = str(pid)
            unit = str(unit)
            if (pid, unit) in unit_map:
                return unit_map[(pid, unit)]
            if pid in base_unit_map and unit == base_unit_map[pid]:
                return 1.0
            return 1.0

        s["factor"] = [
            _factor_lookup(pid, unit)
            for pid, unit in zip(s["product_name"], s["unit"])
        ]
        s["qty_base"] = s["qty"] * s["factor"]

        s["line_no"] = s.groupby("order_id", sort=False).cumcount() + 1

        order_lines = pd.DataFrame({
            "order_id": s["order_id"],
            "line_no": s["line_no"].astype("Int64"),
            "product_id": s["product_name"],
            "product_name": s["product_name"],
            "unit": s["unit"],
            "qty": s["qty"],
            "qty_base": s["qty_base"],
            "unit_price": s["unit_price"],
            "line_amount": s["line_amount"],
            "is_fee": False,
        })

        g = s.groupby("order_id", dropna=False, sort=False)
        orders = pd.DataFrame({
            "order_id": g["order_id"].first(),
            "order_datetime": g["order_datetime"].first(),
            "order_date": g["order_date"].first(),
            "status_raw": g["status_raw"].first(),
            "status": g["status"].first(),
            "customer_name": g["customer_name"].first(),
            "goods_total": g["line_amount"].sum(min_count=1),
        }).reset_index(drop=True)

    tables = {
        "orders": orders,
        "order_lines": order_lines,
    }

    # -------------------------
    # Non-core tables
    # -------------------------
    try:
        if sales.empty and (conv is None or conv.empty):
            products = pd.DataFrame(columns=["product_id", "name", "base_unit"])
        else:
            product_names = []
            if not sales.empty and "Tên sản phẩm" in sales.columns:
                product_names.extend(_clean_str_series(sales["Tên sản phẩm"]).dropna().tolist())
            if conv is not None and not conv.empty and "Tên sản phẩm" in conv.columns:
                product_names.extend(_clean_str_series(conv["Tên sản phẩm"]).dropna().tolist())
            product_names = pd.Index(pd.unique(pd.Series(product_names).dropna()))

            inferred_base = {}
            if not sales.empty and "Tên sản phẩm" in sales.columns and "Đơn vị tính" in sales.columns:
                tmp = sales.rename(columns={"Tên sản phẩm": "product_name", "Đơn vị tính": "unit"}).copy()
                tmp["product_name"] = _clean_str_series(tmp["product_name"])
                tmp["unit"] = _clean_str_series(tmp["unit"])
                for pid, grp in tmp.groupby("product_name", dropna=False):
                    if pd.isna(pid):
                        continue
                    pid = str(pid)
                    if pid in base_unit_map:
                        continue
                    modes = grp["unit"].dropna().mode()
                    if len(modes):
                        inferred_base[pid] = str(modes.iloc[0])

            rows = []
            for pid in product_names:
                pid = str(pid)
                rows.append({
                    "product_id": pid,
                    "name": pid,
                    "base_unit": base_unit_map.get(pid, inferred_base.get(pid, pd.NA)),
                })
            products = pd.DataFrame(rows)

        tables["products"] = products
    except Exception as e:
        meta.setdefault("table_errors", {})["products"] = repr(e)

    try:
        if conv is None or conv.empty:
            unit_conversions = pd.DataFrame(columns=["product_id", "unit", "factor"])
        else:
            c = conv.rename(columns={
                "Tên sản phẩm": "product_name",
                "Đơn vị quy đổi": "unit",
                "Tỉ lệ quy đổi": "factor",
                "Đơn vị cơ bản": "base_unit",
            }).copy()
            c["product_id"] = _clean_str_series(c["product_name"])
            c["unit"] = _clean_str_series(c["unit"])
            c["factor"] = parse_number(c["factor"])
            unit_conversions = c[["product_id", "unit", "factor"]].dropna(subset=["product_id", "unit", "factor"]).reset_index(drop=True)

        tables["unit_conversions"] = unit_conversions
    except Exception as e:
        meta.setdefault("table_errors", {})["unit_conversions"] = repr(e)

    return tables, meta
