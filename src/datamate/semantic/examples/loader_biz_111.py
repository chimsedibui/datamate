def load(raw):
    o = raw.read_table("don_hang.csv")
    orders = pd.DataFrame({
        "order_id": o["Mã chứng từ"],
        "order_datetime": parse_datetime(o["NgayBan"]),
        "status_raw": o["Xử lý"],
        "status": o["Xử lý"].map({"completed": "completed", "cancelled": "cancelled",
                                  "returned": "returned", "shipping": "delivering"}),
        "customer_id": o["Số KH"],
        "channel": o["Bán qua"],
        "payment_method_raw": o["Phương thức TT"],
        "payment_method": o["Phương thức TT"],
        "total_payable": parse_money(o["Tổng bill"]),
        "paid_amount": parse_money(o["Khách trả"]),
        "debt_remaining": parse_money(o["Còn lại"]),
    })
    k = raw.read_table("khach_hang.csv")
    customers = pd.DataFrame({"customer_id": k["ma_kh"], "name": k["Khách lẻ/sỉ"], "group": k["Nhóm"],
                              "phone": k["Phone"]})
    orders = orders.merge(customers.rename(columns={"name": "customer_name", "group": "customer_group"})
                          [["customer_id", "customer_name", "customer_group"]], on="customer_id", how="left")

    p = raw.read_table("san_pham.csv")
    products = pd.DataFrame({"product_id": p["SKU"], "name": p["Sản phẩm"], "category": p["Nhom hang"],
                             "base_unit": p["Đơn vị"], "sell_price": parse_money(p["Giá"])})

    d = raw.read_table("chi_tiet_don.csv")
    q = parse_qty_unit(d["SL"])
    lines = pd.DataFrame({
        "order_id": d["Ma don"], "line_no": parse_number(d["Thứ tự"]), "product_id": d["ID SP"],
        "unit": d["Đơn vị (tính)"], "qty": q["qty"], "qty_base": parse_number(d["SL quy đổi"]),
        "unit_price": parse_money(d["Giá đơn vị"]), "line_amount": parse_money(d["T.Tiền"]),
    })
    lines = lines.merge(products[["product_id", "name"]].rename(columns={"name": "product_name"}),
                        on="product_id", how="left")
    conv = products[products["name"].isin(["Bánh croissant", "Bánh tiramisu"])][["product_id"]].assign(unit="hộp", factor=6.0)

    t = raw.read_table("thanh_toan.csv")
    payments = pd.DataFrame({"payment_id": t["Mã phiếu thu"], "order_id": t["Số HĐ"],
                             "paid_datetime": parse_datetime(t["Thời gian thu"]),
                             "amount": parse_money(t["Tiền"]), "method": t["Phương thức"]})
    meta = {"industry": "cà phê", "has_customer_codes": True, "line_no_from_file": True,
            "data_end": "2026-08-19", "notes": ["order time not recorded on orders (date only)"], "extra_desc": {}}
    return {"orders": orders, "order_lines": lines, "products": products, "unit_conversions": conv,
            "customers": customers, "payments": payments}, meta
