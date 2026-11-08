def load(raw):
    # Helper functions
    def parse_money_column(df, col):
        df[col] = parse_money(df[col])
        return df

    def parse_date_column(df, col):
        df[col] = parse_date(df[col])
        return df

    def parse_datetime_column(df, col):
        df[col] = parse_datetime(df[col])
        return df

    # Load and process each file
    # 1. Orders
    orders_df = raw.read_table("BanHang_HoaDon.xlsx")
    orders_df = parse_money_column(orders_df, "Đã thu")
    orders_df = parse_money_column(orders_df, "Còn nợ")
    orders_df = parse_money_column(orders_df, "Tiền hàng trước giảm")
    orders_df = parse_money_column(orders_df, "Tong tien")
    orders_df = parse_datetime_column(orders_df, "Giờ tạo đơn")
    orders_df = parse_date_column(orders_df, "Ngày bán")
    orders = pd.DataFrame({
        "order_id": orders_df["Order ID"],
        "order_datetime": orders_df["Giờ tạo đơn"],
        "order_date": orders_df["Ngày bán"],
        "status": orders_df["Trạng thái"].map({
            "Đã giao": "completed",
            "Không giao được": "cancelled",
            "Đổi trả": "returned",
            "Đang vận chuyển": "delivering"
        }),
        "status_raw": orders_df["Trạng thái"],
        "customer_id": orders_df["Mã KH mua"].replace("", None),
        "customer_name": orders_df["Tên KH"].replace("Khách lẻ", None),
        "employee": orders_df["Mã NV bán"],
        "branch": None,  # Not recorded
        "channel": orders_df["Nguồn đơn"],
        "payment_method": orders_df["Phương thức TT"].map({
            "Tiền mặt": "cash",
            "Chuyển khoản": "transfer",
            "Ví điện tử": "ewallet",
            "Thu hộ (COD)": "cod",
            "Ghi nợ": "debt"
        }),
        "payment_method_raw": orders_df["Phương thức TT"],
        "goods_total": orders_df["Tiền hàng trước giảm"],
        "order_discount": parse_money(orders_df["CK"]),
        "shipping_fee": None,  # Not recorded
        "vat": None,  # Not recorded
        "total_payable": orders_df["Tong tien"],
        "paid_amount": orders_df["Đã thu"],
        "debt_remaining": orders_df["Còn nợ"],
        "note": None  # Not recorded
    })

    # 2. Order lines
    order_lines_df = raw.read_table("BanHang_ChiTietHoaDon.xlsx")
    order_lines_df = parse_money_column(order_lines_df, "Giá vốn lúc bán")
    order_lines_df = parse_money_column(order_lines_df, "Khuyến mãi")
    order_lines_df = parse_money_column(order_lines_df, "Giá niêm yết")
    order_lines_df = parse_money_column(order_lines_df, "Thanh tien")
    order_lines = pd.DataFrame({
        "order_id": order_lines_df["Mã giao dịch"],
        "line_no": order_lines_df["TT dòng"],
        "product_id": order_lines_df["Ma hang"],
        "product_name": order_lines_df["Chi tiết hàng"],
        "unit": order_lines_df["Đ.vị"],
        "qty": order_lines_df["Slg"],
        "qty_base": order_lines_df["SL cơ bản"],
        "unit_price": order_lines_df["Giá niêm yết"],
        "line_discount": order_lines_df["Khuyến mãi"],
        "line_amount": order_lines_df["Thanh tien"],
        "cost_per_unit": order_lines_df["Giá vốn lúc bán"],
        "is_fee": False  # No service fees recorded
    })

    # 3. Products
    products_df = raw.read_table("BanHang_HangHoa.xlsx")
    products = pd.DataFrame({
        "product_id": products_df["ma_sp"],
        "name": products_df["Món"],
        "aliases": None,  # Not recorded
        "category": products_df["Phân loại"],
        "brand": None,  # Not recorded
        "base_unit": products_df["Đơn vị"],
        "sell_price": products_df["Giá"],
        "cost_price": products_df["Giá vốn hiện tại"],
        "is_prescription": None,  # Not recorded in this file
        "active_ingredient": None,  # Not recorded in this file
        "size": None,  # Not recorded
        "is_prepared": False,  # Not applicable
        "active": True  # All products in the catalogue are active
    })

    # 4. Unit conversions
    conversions_df = raw.read_table("BanHang_DonViQuyDoi.xlsx")
    unit_conversions = pd.DataFrame({
        "product_id": conversions_df["Mã sp"],
        "unit": conversions_df["Đơn vị"],
        "factor": conversions_df["Bằng"]
    })

    # 5. Customers
    customers_df = raw.read_table("BanHang_KhachHang.xlsx")
    customers = pd.DataFrame({
        "customer_id": customers_df["Customer ID"],
        "name": customers_df["ten_kh"],
        "group": customers_df["Hạng"],
        "phone": customers_df["Số điện thoại"],
        "registered_date": None  # Not recorded
    })

    # 6. Payments
    payments_df = raw.read_table("BanHang_PhieuThu.xlsx")
    payments_df = parse_money_column(payments_df, "Số tiền thu")
    payments_df = parse_datetime_column(payments_df, "Thời gian thu")
    payments = pd.DataFrame({
        "payment_id": payments_df["Phiếu thu"],
        "order_id": payments_df["Mã bill"],
        "paid_datetime": payments_df["Thời gian thu"],
        "amount": payments_df["Số tiền thu"],
        "method": payments_df["PTTT"].map({
            "Tiền mặt": "cash",
            "Chuyển khoản": "transfer",
            "Ví điện tử": "ewallet",
            "Thu hộ (COD)": "cod"
        }),
        "kind": "payment"  # All rows are payments
    })

    # 7. Stock
    stock_df = raw.read_table("ton_kho.csv")
    stock = pd.DataFrame({
        "product_id": stock_df["ID SP"],
        "product_name": stock_df["Hàng hóa"],
        "qty_base": stock_df["Ton kho"],
        "as_of_date": "2026-05-25"  # From README
    })

    # 8. Expenses
    expenses_df = raw.read_table("SoTay_ChiPhi.xlsx", skip_rows=2)
    expenses_df = parse_money_column(expenses_df, "So tien")
    expenses_df = parse_date_column(expenses_df, "Ngày chi")
    expenses = pd.DataFrame({
        "expense_date": expenses_df["Ngày chi"],
        "category": expenses_df["Mục"],
        "amount": expenses_df["So tien"],
        "note": expenses_df["Ghi chú"]
    })

    # 9. Purchases
    purchases_df = raw.read_table("SoTay_NhapHang.xlsx", skip_rows=2)
    purchases_df = parse_money_column(purchases_df, "Thành tiền")
    purchases_df = parse_money_column(purchases_df, "Giá nhập")
    purchases_df = parse_date_column(purchases_df, "Ngày nhập")
    purchases = pd.DataFrame({
        "receipt_id": purchases_df["Số phiếu"],
        "receipt_date": purchases_df["Ngày nhập"],
        "supplier": purchases_df["Nhà cung cấp"],
        "status": "ok",  # All rows are valid
        "product_id": None,  # Not recorded
        "product_name": purchases_df["Product"],
        "unit": purchases_df["ĐVT nhập"],
        "qty": purchases_df["SL nhập"],
        "qty_base": None,  # Not recorded
        "unit_cost": purchases_df["Giá nhập"],
        "amount": purchases_df["Thành tiền"],
        "batch_id": purchases_df["Lô"],
        "expiry_date": purchases_df["HSD"],
        "note": purchases_df["Mô tả"]
    })

    # Meta information
    meta = {
        "industry": "nhà thuốc",
        "has_customer_codes": True,
        "line_no_from_file": True,
        "data_end": "2026-05-25",
        "notes": [
            "Debt recorded with remaining amount in 'Còn nợ'",
            "Unit conversions provided in BanHang_DonViQuyDoi.xlsx",
            "Prescription drugs marked in BanHang_Thuoc.xlsx"
        ],
        "extra_desc": {}
    }

    # Return tables and meta
    tables = {
        "orders": orders,
        "order_lines": order_lines,
        "products": products,
        "unit_conversions": unit_conversions,
        "customers": customers,
        "payments": payments,
        "stock": stock,
        "expenses": expenses,
        "purchases": purchases
    }
    return tables, meta
