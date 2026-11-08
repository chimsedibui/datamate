def load(raw):
    import pandas as pd

    meta = {
        "industry": "nhà thuốc",
        "has_customer_codes": True,
        "line_no_from_file": True,
        "data_end": "2026-06-28",
        "notes": [
            "SLQuyDoi is already in base units",
            "GiaVonLucBan is cost price per unit at sale",
            "TongThanhToan includes order-level adjustments",
            "TienHoanTra is refund amount",
            "PhuongThucTT maps to payment methods",
        ],
        "extra_desc": {}
    }

    tables = {}

    # orders
    orders_query = """
        SELECT MaHoaDon AS order_id,
               ThoiGianLap AS order_datetime,
               DATE(ThoiGianLap) AS order_date,
               TinhTrang AS status_raw,
               MaKhachHang AS customer_id,
               NULL AS customer_name,
               NULL AS customer_group,
               MaNhanVien AS employee,
               MaChiNhanh AS branch,
               KenhBanHang AS channel,
               PhuongThucTT AS payment_method_raw,
               TongTienHang AS goods_total,
               ChietKhau AS order_discount,
               PhiVanChuyen AS shipping_fee,
               ThueGTGT AS vat,
               TongThanhToan AS total_payable,
               KhachDaTra AS paid_amount,
               ConPhaiThu AS debt_remaining,
               NULL AS note
        FROM don_hang
    """
    orders = raw.read_sql("pos.sqlite", orders_query)
    orders["status"] = map_status(
        orders["status_raw"],
        {
            "completed": "completed",
            "cancelled": "cancelled",
            "returned": "returned",
            "shipping": "delivering",
        },
    )
    tables["orders"] = orders

    # order_lines
    order_lines_query = """
        SELECT cd.MaHoaDon AS order_id,
               cd.SoThuTu AS line_no,
               cd.MaHangHoa AS product_id,
               sp.TenGoi AS product_name,
               cd.DonViTinh AS unit,
               cd.SL AS qty,
               cd.SLQuyDoi AS qty_base,
               cd.GiaDonVi AS unit_price,
               cd.ChietKhau AS line_discount,
               cd.TriGia AS line_amount,
               cd.GiaVonLucBan AS cost_per_unit
        FROM chi_tiet_don cd
        LEFT JOIN san_pham sp ON cd.MaHangHoa = sp.MaHangHoa
    """
    order_lines = raw.read_sql("pos.sqlite", order_lines_query)
    order_lines["is_fee"] = False
    tables["order_lines"] = order_lines

    # products
    products_query = """
        SELECT sp.MaHangHoa AS product_id,
               sp.TenGoi AS name,
               NULL AS aliases,
               sp.DanhMuc AS category,
               sp.thuong_hieu AS brand,
               sp.DonViCoSo AS base_unit,
               sp.GiaBanLe AS sell_price,
               sp.GiaVonHang AS cost_price,
               tt.BanTheoDon AS is_prescription,
               tt.ThanhPhanChinh AS active_ingredient,
               NULL AS size,
               NULL AS is_prepared,
               sp.ConKinhDoanh AS active
        FROM san_pham sp
        LEFT JOIN thong_tin_thuoc tt ON sp.MaHangHoa = tt.MaHangHoa
    """
    products = raw.read_sql("pos.sqlite", products_query)
    tables["products"] = products

    # unit_conversions
    unit_conversions_query = """
        SELECT MaHangHoa AS product_id,
               DonViTinh AS unit,
               HeSoQuyDoi AS factor
        FROM don_vi_sp
    """
    unit_conversions = raw.read_sql("pos.sqlite", unit_conversions_query)
    tables["unit_conversions"] = unit_conversions

    # customers
    customers_query = """
        SELECT MaKhachHang AS customer_id,
               TenGoi AS name,
               LoaiKhachHang AS "group",
               SoDienThoai AS phone,
               TaoLuc AS registered_date
        FROM khach_hang
    """
    customers = raw.read_sql("pos.sqlite", customers_query)
    tables["customers"] = customers

    # payments
    payments_query = """
        SELECT MaPhieuThu AS payment_id,
               MaHoaDon AS order_id,
               NgayGioThu AS paid_datetime,
               SoTienThu AS amount,
               PhuongThuc AS method,
               'payment' AS kind
        FROM thanh_toan
    """
    payments = raw.read_sql("pos.sqlite", payments_query)
    tables["payments"] = payments

    # purchases
    try:
        purchases_query = """
            SELECT phieu_nhap.MaPhieuNhap AS receipt_id,
                   DATE(phieu_nhap.NgayNhapKho) AS receipt_date,
                   phieu_nhap.MaNhaCungCap AS supplier,
                   phieu_nhap.TinhTrang AS status,
                   chi_tiet_nhap.MaHangHoa AS product_id,
                   sp.TenGoi AS product_name,
                   chi_tiet_nhap.DonViTinh AS unit,
                   chi_tiet_nhap.SL AS qty,
                   lo_hang.SLNhapQuyDoi AS qty_base,
                   chi_tiet_nhap.DonGiaNhap AS unit_cost,
                   phieu_nhap.TongGiaTriNhap AS amount,
                   chi_tiet_nhap.MaLo AS batch_id,
                   lo_hang.HanSuDung AS expiry_date,
                   phieu_nhap.DienGiai AS note
            FROM phieu_nhap
            LEFT JOIN chi_tiet_nhap ON phieu_nhap.MaPhieuNhap = chi_tiet_nhap.MaPhieuNhap
            LEFT JOIN lo_hang ON chi_tiet_nhap.MaLo = lo_hang.MaLo
            LEFT JOIN san_pham sp ON chi_tiet_nhap.MaHangHoa = sp.MaHangHoa
        """
        purchases = raw.read_sql("pos.sqlite", purchases_query)
        purchases["status"] = purchases["status"].map({"received": "ok"})
        tables["purchases"] = purchases
    except Exception as e:
        meta.setdefault("table_errors", {})["purchases"] = repr(e)

    # stock
    try:
        stock_query = """
            SELECT tk.MaHangHoa AS product_id,
                   sp.TenGoi AS product_name,
                   tk.TonCuoi AS qty_base,
                   DATE(tk.NgayGioPhatSinh) AS as_of_date
            FROM the_kho tk
            LEFT JOIN san_pham sp ON tk.MaHangHoa = sp.MaHangHoa
            WHERE tk.NgayGioPhatSinh = (
                SELECT MAX(NgayGioPhatSinh) FROM the_kho
            )
        """
        stock = raw.read_sql("pos.sqlite", stock_query)
        tables["stock"] = stock
    except Exception as e:
        meta.setdefault("table_errors", {})["stock"] = repr(e)

    # batches
    try:
        batches_query = """
            SELECT MaLo AS batch_id,
                   MaHangHoa AS product_id,
                   HanSuDung AS expiry_date,
                   SLNhapQuyDoi AS qty_in_base,
                   DonGiaNhap AS unit_cost_base,
                   MaPhieuNhap AS receipt_id
            FROM lo_hang
        """
        batches = raw.read_sql("pos.sqlite", batches_query)
        tables["batches"] = batches
    except Exception as e:
        meta.setdefault("table_errors", {})["batches"] = repr(e)

    # batch_sales
    try:
        batch_sales_query = """
            SELECT MaHoaDon AS order_id,
                   SoThuTu AS line_no,
                   MaLo AS batch_id,
                   SLQuyDoi AS qty_base,
                   SLTraQuyDoi AS qty_returned_base
            FROM chi_tiet_don_lo
        """
        batch_sales = raw.read_sql("pos.sqlite", batch_sales_query)
        tables["batch_sales"] = batch_sales
    except Exception as e:
        meta.setdefault("table_errors", {})["batch_sales"] = repr(e)

    # price_history
    try:
        price_history_query = """
            SELECT MaHangHoa AS product_id,
                   NgayApDung AS effective_date,
                   GiaBanLe AS price_base,
                   NULL AS old_price_base
        FROM lich_su_gia
    """
        price_history = raw.read_sql("pos.sqlite", price_history_query)
        tables["price_history"] = price_history
    except Exception as e:
        meta.setdefault("table_errors", {})["price_history"] = repr(e)

    # returns
    try:
        returns_query = """
            SELECT MaPhieuTra AS return_id,
                   MaHoaDon AS order_id,
                   NgayGioTra AS return_datetime,
                   TienHoanTra AS refund_amount,
                   NguyenNhan AS reason
            FROM tra_hang
        """
        returns = raw.read_sql("pos.sqlite", returns_query)
        tables["returns"] = returns
    except Exception as e:
        meta.setdefault("table_errors", {})["returns"] = repr(e)

    # deliveries
    try:
        deliveries_query = """
            SELECT MaChungTu AS order_id,
                   DATE(NgayGioPhatSinh) AS delivery_date,
                   LoaiGiaoDich AS status,
                   NULL AS note
            FROM the_kho
            WHERE KieuChungTu = 'order' AND LoaiGiaoDich = 'out_sale'
        """
        deliveries = raw.read_sql("pos.sqlite", deliveries_query)
        deliveries["status"] = deliveries["status"].map({"out_sale": "delivered"})
        tables["deliveries"] = deliveries
    except Exception as e:
        meta.setdefault("table_errors", {})["deliveries"] = repr(e)

    return tables, meta
