def load(raw):
    # Initialize tables and metadata
    tables = {}
    meta = {
        "industry": "tạp hoá",
        "has_customer_codes": True,
        "line_no_from_file": True,
        "data_end": "2026-08-05",
        "notes": [
            "Debt is recorded as individual payment rows in SoQuy.",
            "Unit conversions are provided in DanhSachDonViTinh sheet.",
            "Cost price is recorded per unit sold in DanhSachHoaDonChiTiet sheet.",
            "Stock quantities are recorded in the base unit in BaoCaoTonKho sheet."
        ],
        "extra_desc": {}
    }

    # Load sheets from DuLieuBanHang.xlsx
    try:
        orders_raw = raw.read_table("DuLieuBanHang.xlsx", sheet="DanhSachHoaDon")
        order_lines_raw = raw.read_table("DanhSachHoaDonChiTiet.xlsx", sheet="Sheet1")
        products_raw = raw.read_table("DuLieuBanHang.xlsx", sheet="DanhSachSanPham")
        customers_raw = raw.read_table("DuLieuBanHang.xlsx", sheet="DanhSachKhachHang")
        stock_raw = raw.read_table("DuLieuBanHang.xlsx", sheet="BaoCaoTonKho")
        purchases_raw = raw.read_table("DuLieuBanHang.xlsx", sheet="DanhSachNhapHang")
        purchase_lines_raw = raw.read_table("DuLieuBanHang.xlsx", sheet="DanhSachChiTietNhapHang")
        payments_raw = raw.read_table("DuLieuBanHang.xlsx", sheet="SoQuy")
        unit_conversions_raw = raw.read_table("DuLieuBanHang.xlsx", sheet="DanhSachDonViTinh")
    except Exception as e:
        meta.setdefault("table_errors", {})["load_sheets"] = repr(e)
        return tables, meta

    # Map orders table
    orders = orders_raw.rename(columns={
        "Mã hóa đơn": "order_id",
        "Thời gian tạo": "order_datetime",
        "Tổng tiền hàng": "goods_total",
        "Giảm giá": "order_discount",
        "Tổng sau giảm giá": "total_payable",
        "Khách đã trả": "paid_amount",
        "Phương thức thanh toán": "payment_method_raw",
        "Trạng thái": "status_raw",
        "Mã KH": "customer_id",
        "Khách hàng": "customer_name",
        "Kênh bán": "channel",
        "Người bán": "employee"
    })
    orders["order_datetime"] = parse_datetime(orders["order_datetime"])
    orders["order_date"] = parse_date(orders["order_datetime"])
    orders["status"] = map_status(orders["status_raw"], {
        "Hoàn thành": "completed",
        "đã lấy": "completed",
        "hủy đơn": "cancelled",
        "Đã huỷ": "cancelled",
        "trả hàng": "returned",
        "Đã trả hàng": "returned",
        "Đang xử lý": "delivering"
    })
    orders["debt_remaining"] = parse_money(orders["total_payable"]) - parse_money(orders["paid_amount"])
    orders["goods_total"] = parse_money(orders["goods_total"])
    orders["order_discount"] = parse_money(orders["order_discount"])
    orders["total_payable"] = parse_money(orders["total_payable"])
    orders["paid_amount"] = parse_money(orders["paid_amount"])
    orders["payment_method"] = map_status(orders["payment_method_raw"], {
        "Tiền mặt": "cash",
        "Chuyển khoản": "transfer",
        "Ví điện tử": "ewallet",
        "Thu hộ (COD)": "cod",
        "Ghi nợ": "debt"
    })
    orders["customer_id"] = orders["customer_id"].replace({"kl": None, "lẻ": None, "khách lẻ": None})
    orders["customer_name"] = orders["customer_name"].replace({"kl": None, "lẻ": None, "khách lẻ": None})
    tables["orders"] = orders[[
        "order_id", "order_datetime", "order_date", "status", "status_raw", "customer_id", "customer_name",
        "employee", "channel", "payment_method", "payment_method_raw", "goods_total", "order_discount",
        "total_payable", "paid_amount", "debt_remaining"
    ]]

    # Map order_lines table
    order_lines = order_lines_raw.rename(columns={
        "Mã hóa đơn": "order_id",
        "STT": "line_no",
        "Mã hàng": "product_id",
        "Tên hàng": "product_name",
        "Đơn vị tính": "unit",
        "Số lượng": "qty",
        "Đơn giá": "unit_price",
        "Giảm giá": "line_discount",
        "Thành tiền": "line_amount",
        "Số lượng quy đổi": "qty_base",
        "Giá vốn": "cost_per_unit"
    })
    order_lines["qty"] = parse_number(order_lines["qty"])
    order_lines["unit_price"] = parse_money(order_lines["unit_price"])
    order_lines["line_discount"] = parse_money(order_lines["line_discount"])
    order_lines["line_amount"] = parse_money(order_lines["line_amount"])
    order_lines["qty_base"] = parse_number(order_lines["qty_base"])
    order_lines["cost_per_unit"] = parse_money(order_lines["cost_per_unit"])
    tables["order_lines"] = order_lines[[
        "order_id", "line_no", "product_id", "product_name", "unit", "qty", "qty_base", "unit_price",
        "line_discount", "line_amount", "cost_per_unit"
    ]]

    # Map products table
    products = products_raw.rename(columns={
        "Mã hàng": "product_id",
        "Tên hàng": "name",
        "Nhóm hàng(3 Cấp)": "category",
        "ĐVT": "base_unit",
        "Giá bán": "sell_price",
        "Giá vốn": "cost_price",
        "Đang kinh doanh": "active"
    })
    products["sell_price"] = parse_money(products["sell_price"])
    products["cost_price"] = parse_money(products["cost_price"])
    products["active"] = products["active"].astype(bool)
    tables["products"] = products[[
        "product_id", "name", "category", "base_unit", "sell_price", "cost_price", "active"
    ]]

    # Map customers table
    customers = customers_raw.rename(columns={
        "Mã khách hàng": "customer_id",
        "Tên khách hàng": "name",
        "Nhóm khách hàng": "group",
        "Điện thoại": "phone",
        "Ngày tạo": "registered_date"
    })
    customers["registered_date"] = parse_date(customers["registered_date"])
    tables["customers"] = customers[["customer_id", "name", "group", "phone", "registered_date"]]

    # Map stock table
    stock = stock_raw.rename(columns={
        "Mã hàng": "product_id",
        "Tên hàng": "product_name",
        "ĐVT": "base_unit",
        "Tồn cuối kỳ": "qty_base"
    })
    stock["qty_base"] = parse_number(stock["qty_base"])
    stock["as_of_date"] = pd.to_datetime(meta["data_end"]).date()
    tables["stock"] = stock[["product_id", "product_name", "qty_base", "as_of_date"]]

    # Map purchases table
    try:
        purchases = purchases_raw.rename(columns={
            "Mã nhập hàng": "receipt_id",
            "Thời gian": "receipt_date",
            "Mã NCC": "supplier",
            "Cần trả NCC": "amount",
            "Trạng thái": "status",
            "Ghi chú": "note"
        })
        purchases["receipt_date"] = parse_date(purchases["receipt_date"])
        purchases["amount"] = parse_money(purchases["amount"])
        purchases["status"] = purchases["status"].map({"Đã nhập hàng": "ok"})
        tables["purchases"] = purchases[["receipt_id", "receipt_date", "supplier", "amount", "status", "note"]]
    except Exception as e:
        meta.setdefault("table_errors", {})["purchases"] = repr(e)

    # Map purchase_lines table
    try:
        purchase_lines = purchase_lines_raw.rename(columns={
            "Mã nhập hàng": "receipt_id",
            "STT": "line_no",
            "Mã hàng": "product_id",
            "ĐVT": "unit",
            "Số lượng": "qty",
            "Đơn giá": "unit_cost"
        })
        purchase_lines["qty"] = parse_number(purchase_lines["qty"])
        purchase_lines["unit_cost"] = parse_money(purchase_lines["unit_cost"])
        purchase_lines["amount"] = purchase_lines["qty"] * purchase_lines["unit_cost"]
        tables["purchase_lines"] = purchase_lines[[
            "receipt_id", "line_no", "product_id", "unit", "qty", "unit_cost", "amount"
        ]]
    except Exception as e:
        meta.setdefault("table_errors", {})["purchase_lines"] = repr(e)

    # Map payments table
    try:
        payments = payments_raw.rename(columns={
            "Mã phiếu": "payment_id",
            "Thời gian": "paid_datetime",
            "Mã hóa đơn": "order_id",
            "Phương thức thanh toán": "method",
            "Giá trị": "amount"
        })
        payments["paid_datetime"] = parse_datetime(payments["paid_datetime"])
        payments["amount"] = parse_money(payments["amount"])
        payments["method"] = map_status(payments["method"], {
            "Tiền mặt": "cash",
            "Chuyển khoản": "transfer",
            "Ví điện tử": "ewallet",
            "Thu hộ (COD)": "cod"
        })
        payments["kind"] = "payment"
        tables["payments"] = payments[["payment_id", "order_id", "paid_datetime", "amount", "method", "kind"]]
    except Exception as e:
        meta.setdefault("table_errors", {})["payments"] = repr(e)

    # Map unit_conversions table
    try:
        unit_conversions = unit_conversions_raw.rename(columns={
            "Mã hàng": "product_id",
            "ĐVT": "unit",
            "Quy đổi": "factor"
        })
        unit_conversions["factor"] = parse_number(unit_conversions["factor"])
        tables["unit_conversions"] = unit_conversions[["product_id", "unit", "factor"]]
    except Exception as e:
        meta.setdefault("table_errors", {})["unit_conversions"] = repr(e)

    return tables, meta
