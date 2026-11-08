def load(raw):
    import pandas as pd
    import numpy as np
    from datetime import datetime

    # Read product catalog and unit conversions
    df_danhmuc = raw.read_table('DanhMuc.xlsx', sheet='Sheet1', header_rows=1)
    products_df = df_danhmuc[['Tên món', 'Đơn vị gốc', 'ĐV nhập', 'Hệ số']].copy()
    products_df.rename(columns={'Tên món': 'name', 'Đơn vị gốc': 'base_unit', 'ĐV nhập': 'unit', 'Hệ số': 'factor'}, inplace=True)
    products_df['product_id'] = products_df['name']
    
    # Move unit conversions to separate table
    unit_conversions_df = products_df[['product_id', 'unit', 'factor']].copy()
    products_df = products_df[['product_id', 'name', 'base_unit']]

    # Read sales data
    so_ban_hang_files = ['SoBanHang_1.xlsx', 'SoBanHang_2026.xlsx']
    so_ban_hang_dfs = []
    for file in so_ban_hang_files:
        sheets = raw.sheet_names(file)
        for sheet in sheets:
            df = raw.read_table(file, sheet=sheet, header_rows=1)
            sheet_name = sheet
            year = year_from_text(sheet_name)
            df['Ngày giao dịch'] = parse_date(df['Ngày giao dịch'], default_year=year)
            df['Giờ'] = parse_datetime(df['Giờ'], default_year=year)
            so_ban_hang_dfs.append(df)
    so_ban_hang = pd.concat(so_ban_hang_dfs, ignore_index=True)

    # Process orders and order_lines
    order_lines_df = so_ban_hang[['Mã hóa đơn', 'Hàng', 'so_luong', 'Unit', 'Trị giá', 'Giá lẻ']].copy()
    order_lines_df.rename(columns={'Mã hóa đơn': 'order_id', 'Hàng': 'product_name', 'so_luong': 'qty', 'Unit': 'unit', 'Trị giá': 'line_amount', 'Giá lẻ': 'unit_price'}, inplace=True)
    order_lines_df['qty'] = parse_number(order_lines_df['qty'])
    order_lines_df['unit_price'] = parse_money(order_lines_df['unit_price'])
    order_lines_df['line_amount'] = parse_money(order_lines_df['line_amount'])

    # Filter out non-product lines (notes, totals)
    order_lines_df = order_lines_df[~order_lines_df['product_name'].str.contains('Ghi chú|Cộng', na=False)]

    # Merge with unit conversions
    order_lines_df = order_lines_df.merge(unit_conversions_df, left_on=['product_name', 'unit'], right_on=['product_id', 'unit'], how='left')
    order_lines_df['factor'] = order_lines_df['factor'].fillna(1)
    order_lines_df['qty_base'] = order_lines_df['qty'] * order_lines_df['factor']

    # Build orders
    orders_df = so_ban_hang[['Mã hóa đơn', 'Ngày giao dịch', 'Status', 'Khách', 'Giờ']].copy()
    orders_df.rename(columns={'Mã hóa đơn': 'order_id', 'Ngày giao dịch': 'order_date', 'Status': 'status_raw', 'Khách': 'customer_name', 'Giờ': 'order_datetime'}, inplace=True)
    orders_df['order_date'] = parse_date(orders_df['order_date'])
    orders_df['order_datetime'] = parse_datetime(orders_df['order_datetime'])
    orders_df['customer_id'] = orders_df['customer_name'].apply(lambda x: None if x == 'Khách lẻ' else x)
    orders_df['customer_name'] = orders_df['customer_name'].apply(lambda x: None if x == 'Khách lẻ' else x)

    # Map status
    status_mapping = {
        'đã lấy': 'completed',
        'đã xong': 'completed',
        'hủy': 'cancelled',
        'Đã hủy đơn': 'cancelled',
        'Đổi trả': 'returned',
        'Trả hàng': 'returned',
        'hàng trả về': 'returned',
        'Hoàn trả': 'returned',
    }
    orders_df['status'] = orders_df['status_raw'].apply(lambda x: map_status(x, status_mapping) if pd.notna(x) else None)
    orders_df['status'] = orders_df['status'].fillna('completed')  # Default to completed if status_raw is missing

    # Check for returns
    orders_df['has_return'] = orders_df['order_id'].apply(lambda x: any(order_lines_df[order_lines_df['order_id'] == x]['qty'] < 0))
    orders_df['status'] = orders_df.apply(lambda row: 'returned' if row['has_return'] else row['status'], axis=1)

    # Ensure unique order_id by using groupby and first() to deduplicate
    orders_df = orders_df.groupby('order_id').first().reset_index()

    # Build products
    products_df['category'] = None
    products_df['brand'] = None
    products_df['sell_price'] = None
    products_df['cost_price'] = None
    products_df['is_prescription'] = False
    products_df['active_ingredient'] = None
    products_df['size'] = None
    products_df['is_prepared'] = False
    products_df['active'] = True

    # Read inventory
    ton_kho_df = raw.read_table('TonKho.xlsx', sheet='Sheet1', header_rows=1)
    ton_kho_df.rename(columns={'Hàng hóa': 'product_name', 'Đơn vị tính': 'base_unit', 'Tồn cuối': 'qty_base'}, inplace=True)
    ton_kho_df['product_id'] = ton_kho_df['product_name']
    ton_kho_df['as_of_date'] = pd.to_datetime('2026-06-24')

    # Read purchases
    so_nhap_hang_df = raw.read_table('SoNhapHang.xlsx', sheet='Sheet1', header_rows=1)
    so_nhap_hang_df.rename(columns={'Số phiếu': 'receipt_id', 'Ngày nhập': 'receipt_date', 'Ghi chu': 'note', 'Thành tiền': 'amount', 'Nhà cung cấp': 'supplier', 'Tên mặt hàng': 'product_name', 'Giá nhập': 'unit_cost', 'ĐVT': 'unit', 'SL nhập': 'qty'}, inplace=True)
    so_nhap_hang_df['receipt_date'] = parse_date(so_nhap_hang_df['receipt_date'])
    so_nhap_hang_df['unit_cost'] = parse_money(so_nhap_hang_df['unit_cost'])
    so_nhap_hang_df['qty'] = parse_number(so_nhap_hang_df['qty'])

    # Merge with unit conversions
    so_nhap_hang_df = so_nhap_hang_df.merge(unit_conversions_df, left_on=['product_name', 'unit'], right_on=['product_id', 'unit'], how='left')
    so_nhap_hang_df['factor'] = so_nhap_hang_df['factor'].fillna(1)
    so_nhap_hang_df['qty_base'] = so_nhap_hang_df['qty'] * so_nhap_hang_df['factor']

    # Read price history
    bang_gia_df = raw.read_table('BangGia.xlsx', sheet='Sheet1', header_rows=1)
    bang_gia_df.rename(columns={'ten_hang': 'product_name', 'Ngày áp dụng': 'effective_date', 'Giá mới': 'price_base', 'Giá cũ': 'old_price_base'}, inplace=True)
    bang_gia_df['effective_date'] = parse_date(bang_gia_df['effective_date'])
    bang_gia_df['product_id'] = bang_gia_df['product_name']
    bang_gia_df['price_base'] = parse_money(bang_gia_df['price_base'])
    bang_gia_df['old_price_base'] = parse_money(bang_gia_df['old_price_base'])

    # Build tables
    tables = {
        'orders': orders_df[['order_id', 'order_datetime', 'order_date', 'status', 'status_raw', 'customer_id', 'customer_name']],
        'order_lines': order_lines_df[['order_id', 'product_name', 'unit', 'qty', 'qty_base', 'unit_price', 'line_amount']],
        'products': products_df[['product_id', 'name', 'base_unit']],
        'unit_conversions': unit_conversions_df[['product_id', 'unit', 'factor']],
        'stock': ton_kho_df[['product_id', 'product_name', 'qty_base', 'as_of_date']],
        'purchases': so_nhap_hang_df[['receipt_id', 'receipt_date', 'supplier', 'product_name', 'unit', 'qty', 'qty_base', 'unit_cost', 'amount', 'note']],
        'price_history': bang_gia_df[['product_id', 'effective_date', 'price_base', 'old_price_base']],
    }

    # Meta information
    meta = {
        'industry': 'cà phê',
        'has_customer_codes': False,
        'line_no_from_file': False,
        'data_end': '2026-06-24',
        'notes': [
            'Trả hàng: hàng trả ghi số lượng âm, cùng mã đơn với đơn đã bán',
            'Tồn kho: Tồn cuối là tồn cuối ngày cuối có dữ liệu, theo đơn vị cơ bản',
            'Nhập hàng: Giá nhập là giá nhập một đơn vị, theo đơn vị nhập của dòng; Thành tiền là tiền nhập của cả dòng',
            'Giá bán: Giá mới áp dụng từ Ngày áp dụng đến trước lần đổi giá sau của cùng mặt hàng; dòng đầu mỗi mặt hàng là giá lúc bắt đầu dữ liệu (Giá cũ trống)',
            'Quy đổi đơn vị: số lượng ghi theo đơn vị của dòng, hệ số quy đổi ở DanhMuc.xlsx'
        ],
        'extra_desc': {}
    }

    return tables, meta
