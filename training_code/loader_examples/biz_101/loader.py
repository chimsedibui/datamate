def load(raw):
    import pandas as pd
    import numpy as np
    import re

    meta = {
        "industry": "tạp hoá",
        "has_customer_codes": False,
        "line_no_from_file": False,
        "data_end": "2026-08-24",
        "notes": [
            "File bán hàng dạng phẳng, mỗi dòng là một dòng hàng.",
            "Mã hóa đơn dùng làm order_id; ngày bán không có giờ nên order_datetime được chuẩn hoá về 00:00:00.",
            "Cột Quy ra ĐV lẻ được dùng làm qty_base trực tiếp.",
            "Kết quả có các trạng thái Xong / Đã hủy / Đổi trả / Đang giao; đơn có Đổi trả được gán status returned.",
            "Không có cột phương thức thanh toán; payment_method được để other cho toàn bộ đơn.",
            "Khách lẻ được coi là khách vãng lai; customer_id không có trong dữ liệu.",
        ],
        "extra_desc": {},
        "table_errors": {},
    }

    # ---------- helpers ----------
    def _clean_series(s):
        return s.astype("string").str.strip()

    def _to_num(s):
        return pd.to_numeric(
            s.astype("string").str.replace(",", "", regex=False),
            errors="coerce",
        )

    def _first_nonnull(series):
        for x in series:
            if pd.notna(x):
                return x
        return pd.NA

    def _join_unique(series):
        vals = []
        for x in series:
            if pd.notna(x):
                x = str(x).strip()
                if x and x not in vals:
                    vals.append(x)
        return "|".join(vals) if vals else pd.NA

    def _fold(x):
        if pd.isna(x):
            return ""
        try:
            return fold_text(str(x))
        except Exception:
            s = str(x).lower().strip()
            s = re.sub(r"\s+", " ", s)
            return s

    def _map_status(raw):
        if pd.isna(raw):
            return pd.NA
        txt = str(raw).strip()
        low = txt.lower()
        if any(k in low for k in ["hủy", "huỷ", "cancel"]):
            return "cancelled"
        if any(k in low for k in ["trả hàng", "hoàn trả", "đổi trả"]):
            return "returned"
        if any(k in low for k in ["đang giao", "chờ giao"]):
            return "delivering"
        if low in {"xong", "đã xong", "hoàn thành", "ok"}:
            return "completed"
        return pd.NA

    def _base_unit_lookup():
        # canonical base units, matched by fold_text to survive spelling/spacing variants
        return {
            "Coca Cola 330ml": "lon",
            "Pepsi 330ml": "lon",
            "Sting dâu 330ml": "chai",
            "Red Bull 250ml": "lon",
            "Trà xanh 0 độ 455ml": "chai",
            "Nước suối Aquafina 500ml": "chai",
            "Nước suối Lavie 1.5L": "chai",
            "Bia Tiger 330ml": "lon",
            "Bia Sài Gòn Special 330ml": "lon",
            "Bia Heineken 330ml": "lon",
            "Sữa tươi Vinamilk 180ml": "hộp",
            "Sữa TH true MILK 180ml": "hộp",
            "Sữa đặc Ông Thọ 380g": "hộp",
            "Sữa chua Vinamilk có đường": "hộp",
            "Yakult 65ml": "chai",
            "Mì Hảo Hảo tôm chua cay": "gói",
            "Mì Omachi sườn hầm": "gói",
            "Mì 3 Miền tôm hùm": "gói",
            "Phở Vifon bò": "gói",
            "Gạo ST25": "kg",
            "Gạo thơm Jasmine": "kg",
            "Đường Biên Hòa 1kg": "gói",
            "Muối I-ốt 500g": "gói",
            "Bột ngọt Ajinomoto 400g": "gói",
            "Hạt nêm Knorr 400g": "gói",
            "Nước mắm Nam Ngư 750ml": "chai",
            "Nước tương Maggi 700ml": "chai",
            "Dầu ăn Tường An 1L": "chai",
            "Dầu ăn Simply 2L": "chai",
            "Tương ớt Chinsu 250g": "chai",
            "Trứng gà so": "quả",
            "Xúc xích Vissan 5 cây": "gói",
            "Cá hộp 3 Cô Gái 155g": "hộp",
            "Đậu phộng rang muối 200g": "gói",
            "Snack Oishi tôm cay 40g": "gói",
            "Snack Lay's vị tự nhiên 56g": "gói",
            "Bánh Oreo 133g": "gói",
            "Bánh Chocopie 12 cái": "hộp",
            "Bánh quy Cosy 200g": "gói",
            "Kẹo Alpenliebe 100g": "gói",
            "Kẹo cao su Doublemint": "vỉ",
            "Hạt hướng dương 200g": "gói",
            "Bột giặt OMO 3kg": "gói",
            "Nước giặt Ariel 3.2L": "chai",
            "Nước rửa chén Sunlight 750g": "chai",
            "Nước lau sàn Sunlight 1L": "chai",
            "Kem đánh răng P/S 180g": "hộp",
            "Bàn chải Colgate": "cái",
            "Dầu gội Clear 630g": "chai",
            "Sữa tắm Lifebuoy 850g": "chai",
            "Xà bông Lifebuoy 90g": "cục",
            "Giấy vệ sinh Pulppy 10 cuộn": "lốc",
            "Khăn giấy Tempo 10 gói": "lốc",
            "Băng vệ sinh Diana 8 miếng": "gói",
            "Tã Bobby M 32 miếng": "gói",
            "Thuốc lá Vinataba": "gói",
            "Thuốc lá Jet": "gói",
            "Pin Panasonic AA": "vỉ",
            "Nhang Quảng Trí bó": "bó",
        }

    def _unit_conversion_rows(df):
        # Use observed unit factors from the data where available; keep only non-1 factors
        out = (
            df.loc[df["qty"].notna() & df["qty_base"].notna() & (df["qty"] != 0)]
              .assign(factor=lambda x: x["qty_base"] / x["qty"])
              .groupby(["product_id", "unit"], as_index=False)["factor"]
              .median()
        )
        out = out.loc[(out["factor"].notna()) & (out["factor"] != 1)].copy()
        out["factor"] = out["factor"].astype(float)
        return out

    # ---------- read sales ----------
    df = raw.read_table("ban_hang.csv", header_rows=1)

    df = df.rename(
        columns={
            "Giá bán": "unit_price",
            "Kết quả": "status_raw",
            "Nội dung": "product_name",
            "Ngày tháng": "order_date_raw",
            "Đơn vị (tính)": "unit",
            "so_luong": "qty",
            "Thanh tien": "line_amount",
            "Quy ra ĐV lẻ": "qty_base",
            "Mã hóa đơn": "order_id",
            "Đối tác": "customer_name",
        }
    )

    # clean string columns
    for c in ["status_raw", "product_name", "unit", "order_id", "customer_name"]:
        df[c] = _clean_series(df[c])

    # numeric columns
    for c in ["unit_price", "qty", "line_amount", "qty_base"]:
        df[c] = _to_num(df[c])

    # parse datetime properly
    try:
        df["order_datetime"] = parse_datetime(df["order_date_raw"])
    except Exception:
        df["order_datetime"] = pd.to_datetime(df["order_date_raw"], dayfirst=True, errors="coerce")

    # normalize order date
    df["order_date"] = df["order_datetime"].dt.date

    # product id: keep exact written product name from file
    df["product_id"] = df["product_name"]

    # customers: walk-in -> null
    walkin = df["customer_name"].isna() | df["customer_name"].isin(
        ["", "Khách lẻ", "khách lẻ", "Khách vãng lai", "khách vãng lai", "Vãng lai", "vãng lai"]
    )
    df.loc[walkin, "customer_name"] = pd.NA
    df["customer_id"] = pd.NA

    # status
    df["status"] = df["status_raw"].map(_map_status)

    # any returned line => returned order
    returned_orders = df.loc[df["status"].eq("returned"), "order_id"].dropna().unique()
    if len(returned_orders):
        df.loc[df["order_id"].isin(returned_orders), "status"] = "returned"

    # line number, order lines
    df["line_no"] = df.groupby("order_id", sort=False).cumcount() + 1
    df["line_discount"] = 0.0
    df["cost_per_unit"] = pd.NA
    df["is_fee"] = False

    order_lines = df[
        [
            "order_id",
            "line_no",
            "product_id",
            "product_name",
            "unit",
            "qty",
            "qty_base",
            "unit_price",
            "line_discount",
            "line_amount",
            "cost_per_unit",
            "is_fee",
        ]
    ].copy()

    # orders
    def _agg_status(s):
        s = pd.Series(s).dropna().astype(str)
        if (s == "returned").any():
            return "returned"
        if (s == "cancelled").any():
            return "cancelled"
        if (s == "delivering").any():
            return "delivering"
        if (s == "completed").any():
            return "completed"
        return pd.NA

    orders = (
        df.sort_values(["order_id", "order_datetime"], kind="stable")
          .groupby("order_id", as_index=False)
          .agg(
              order_datetime=("order_datetime", "min"),
              status=("status", _agg_status),
              status_raw=("status_raw", _join_unique),
              customer_name=("customer_name", _first_nonnull),
              goods_total=("line_amount", "sum"),
          )
    )

    orders["order_date"] = orders["order_datetime"].dt.date
    orders["customer_id"] = pd.NA
    orders["customer_group"] = pd.NA
    orders["employee"] = pd.NA
    orders["branch"] = pd.NA
    orders["channel"] = pd.NA
    orders["payment_method"] = "other"
    orders["payment_method_raw"] = pd.NA
    orders["order_discount"] = pd.NA
    orders["shipping_fee"] = pd.NA
    orders["vat"] = pd.NA
    orders["total_payable"] = pd.NA
    orders["paid_amount"] = pd.NA
    orders["debt_remaining"] = pd.NA
    orders["note"] = pd.NA

    orders = orders[
        [
            "order_id",
            "order_datetime",
            "order_date",
            "status",
            "status_raw",
            "customer_id",
            "customer_name",
            "customer_group",
            "employee",
            "branch",
            "channel",
            "payment_method",
            "payment_method_raw",
            "goods_total",
            "order_discount",
            "shipping_fee",
            "vat",
            "total_payable",
            "paid_amount",
            "debt_remaining",
            "note",
        ]
    ]

    tables = {
        "orders": orders,
        "order_lines": order_lines,
    }

    # ---------- non-core tables ----------
    try:
        base_lookup = _base_unit_lookup()
        folded_lookup = {_fold(k): v for k, v in base_lookup.items()}

        # unique products from observed data, preserving appearance order
        observed_products = pd.Series(df["product_id"].dropna().astype(str).unique()).tolist()

        products = pd.DataFrame(
            [
                {
                    "product_id": p,
                    "name": p,
                    "aliases": pd.NA,
                    "category": pd.NA,
                    "brand": pd.NA,
                    "base_unit": folded_lookup.get(_fold(p), pd.NA),
                    "sell_price": pd.NA,
                    "cost_price": pd.NA,
                    "is_prescription": False,
                    "active_ingredient": pd.NA,
                    "size": pd.NA,
                    "is_prepared": False,
                    "active": True,
                }
                for p in observed_products
            ]
        )
        tables["products"] = products
    except Exception as e:
        meta.setdefault("table_errors", {})["products"] = repr(e)

    try:
        tables["unit_conversions"] = _unit_conversion_rows(df)
    except Exception as e:
        meta.setdefault("table_errors", {})["unit_conversions"] = repr(e)

    return tables, meta
