def load(raw):
    # Helper functions
    def parse_vnd(s):
        return parse_money(s.astype(str).str.replace(",", "").str.replace(".", "").str.strip())

    def parse_date_ddmmyyyy(s):
        return pd.to_datetime(s, format='%d/%m/%Y', errors='coerce')

    def parse_datetime_ddmmyyyy_hhmmss(s):
        return pd.to_datetime(s, format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # Load and process orders
    orders_files = raw.list_files()
    orders_data = []
    for file in orders_files:
        if "DanhSachHoaDon.xlsx" in file:
            for sheet in raw.sheet_names(file):
                df = raw.read_table(file, sheet=sheet)
                df['Thời gian'] = parse_date_ddmmyyyy(df['Thời gian'])
                df['Thời gian tạo'] = parse_datetime_ddmmyyyy_hhmmss(df['Thời gian tạo'])
                df['Tổng tiền hàng'] = parse_vnd(df['Tổng tiền hàng'])
                df['Giảm giá'] = parse_vnd(df['Giảm giá'])
                df['Khách đã trả'] = parse_vnd(df['Khách đã trả'])
                df['Tổng sau giảm giá'] = parse_vnd(df['Tổng sau giảm giá'])
                df['Còn nợ'] = parse_vnd(df['Còn nợ'])
                df['status'] = df['Trạng thái'].map({
                    'xong': 'completed',
                    'hủy': 'cancelled',
                    'hàng trả về': 'returned'
                }).fillna('delivering')
                df['customer_id'] = df['Mã KH'].where(df['Mã KH'].notna(), None)
                df['customer_name'] = df['Khách hàng'].where(df['Khách hàng'] != 'Khách lẻ', None)
                df['order_date'] = df['Thời gian']
                df['order_datetime'] = df['Thời gian tạo']
                df['status_raw'] = df['Trạng thái']
                df['goods_total'] = df['Tổng tiền hàng']
                df['order_discount'] = df['Giảm giá']
                df['total_payable'] = df['Tổng sau giảm giá']
                df['paid_amount'] = df['Khách đã trả']
                df['debt_remaining'] = df['Còn nợ']
                orders_data.append(df[[
                    'Mã hóa đơn', 'order_datetime', 'order_date', 'status', 'status_raw',
                    'customer_id', 'customer_name', 'goods_total', 'order_discount',
                    'total_payable', 'paid_amount', 'debt_remaining'
                ]].rename(columns={
                    'Mã hóa đơn': 'order_id'
                }))

    orders = pd.concat(orders_data, ignore_index=True)

    # Load and process order_lines
    order_lines = raw.read_table("DanhSachHoaDonChiTiet.xlsx")
    order_lines['Số lượng'] = parse_number(order_lines['Số lượng'])
    order_lines['Đơn giá'] = parse_vnd(order_lines['Đơn giá'])
    order_lines['Giảm giá'] = parse_vnd(order_lines['Giảm giá'])
    order_lines['Thành tiền'] = parse_vnd(order_lines['Thành tiền'])
    order_lines['Số lượng quy đổi'] = parse_number(order_lines['Số lượng quy đổi'])
    order_lines['is_fee'] = order_lines['Tên hàng'].str.contains('Phí phục vụ', case=False)
    order_lines = order_lines.rename(columns={
        'Mã hóa đơn': 'order_id',
        'Mã hàng': 'product_id',
        'Tên hàng': 'product_name',
        'Số lượng': 'qty',
        'Đơn giá': 'unit_price',
        'Giảm giá': 'line_discount',
        'Thành tiền': 'line_amount',
        'Số lượng quy đổi': 'qty_base',
        'STT': 'line_no'
    })

    # Load and process products
    products = raw.read_table("DanhSachSanPham.xlsx")
    products['Giá bán'] = parse_vnd(products['Giá bán'])
    products = products.rename(columns={
        'Mã hàng': 'product_id',
        'Tên hàng': 'name',
        'Nhóm hàng(3 Cấp)': 'category',
        'Giá bán': 'sell_price',
        'ĐVT': 'base_unit'
    })
    products['active'] = True
    products = products[['product_id', 'name', 'category', 'sell_price', 'base_unit', 'active']]

    # Load and process customers
    customers = raw.read_table("DanhSachKhachHang.xlsx")
    customers = customers.rename(columns={
        'Mã khách hàng': 'customer_id',
        'Tên khách hàng': 'name',
        'Nhóm khách hàng': 'group',
        'Điện thoại': 'phone'
    })
    customers = customers[['customer_id', 'name', 'group', 'phone']]

    # Load and process stock
    stock = raw.read_table("BaoCaoTonKho.xlsx")
    stock['Tồn cuối kỳ'] = parse_number(stock['Tồn cuối kỳ'])
    stock = stock.rename(columns={
        'Mã hàng': 'product_id',
        'Tên hàng': 'product_name',
        'Tồn cuối kỳ': 'qty_base'
    })
    stock['as_of_date'] = pd.Timestamp('2026-09-07')
    stock = stock[['product_id', 'product_name', 'qty_base', 'as_of_date']]

    # Load and process purchases
    purchases = raw.read_table("DanhSachNhapHang.xlsx")
    purchases['Thời gian'] = parse_date_ddmmyyyy(purchases['Thời gian'])
    purchases['Cần trả NCC'] = parse_vnd(purchases['Cần trả NCC'])
    purchases = purchases.rename(columns={
        'Mã nhập hàng': 'receipt_id',
        'Thời gian': 'receipt_date',
        'Mã NCC': 'supplier',
        'Cần trả NCC': 'amount',
        'Trạng thái': 'status',
        'Ghi chú': 'note'
    })
    purchases = purchases[purchases['status'] == 'Đã nhập hàng']

    purchase_lines = raw.read_table("DanhSachChiTietNhapHang.xlsx")
    purchase_lines['Số lượng'] = parse_number(purchase_lines['Số lượng'])
    purchase_lines['Đơn giá'] = parse_vnd(purchase_lines['Đơn giá'])
    purchase_lines = purchase_lines.rename(columns={
        'Mã nhập hàng': 'receipt_id',
        'STT': 'line_no',
        'Mã hàng': 'product_id',
        'ĐVT': 'unit',
        'Số lượng': 'qty',
        'Đơn giá': 'unit_cost'
    })
    purchase_lines = purchase_lines.merge(
        products[['product_id', 'name']],
        on='product_id',
        how='left'
    ).rename(columns={'name': 'product_name'})
    purchases = purchases.merge(
        purchase_lines,
        on='receipt_id',
        how='left'
    )
    purchases = purchases[['receipt_id', 'receipt_date', 'supplier', 'product_id', 'product_name', 'unit', 'qty', 'unit_cost', 'amount', 'note']]

    # Load and process unit conversions
    unit_conversions_data = [
        ('SP0002', 'thùng', 24),
        ('SP0006', 'thùng', 24),
        ('SP0030', 'két', 20),
        ('SP0007', 'thùng', 24),
        ('SP0008', 'thùng', 24),
        ('SP0009', 'thùng', 24),
        ('SP0010', 'thùng', 24),
    ]
    unit_conversions = pd.DataFrame(unit_conversions_data, columns=['product_id', 'unit', 'factor'])

    # Prepare metadata
    meta = {
        'industry': 'nhà hàng',
        'has_customer_codes': True,
        'line_no_from_file': True,
        'data_end': '2026-09-07',
        'notes': [
            'Units have specific conversion factors for some products',
            'Debt is recorded as "Còn nợ" in orders',
            'Service fee lines are marked as "Phí phục vụ"'
        ],
        'extra_desc': {}
    }

    # Return tables and metadata
    tables = {
        'orders': orders,
        'order_lines': order_lines,
        'products': products,
        'customers': customers,
        'stock': stock,
        'purchases': purchases,
        'unit_conversions': unit_conversions
    }
    return tables, meta
