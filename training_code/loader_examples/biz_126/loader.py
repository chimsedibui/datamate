def load(raw):
    # Read the raw data from the Excel file
    file = 'SoBanHang.xlsx'
    sheet = 'Sheet1'
    df = raw.read_table(file, sheet=sheet)

    # Parse and clean the data
    df['order_datetime'] = parse_datetime(df['Ngày tháng'])
    df['order_date'] = df['order_datetime'].dt.date
    df['status_raw'] = df['Trạng thái đơn']
    df['status'] = map_status(df['status_raw'], {
        'Đã giao': 'completed',
        'hủy': 'cancelled',
        'Trả hàng': 'returned',
        'đang gửi': 'delivering'
    })
    df['line_amount'] = parse_money(df['Thành tiền (đ)'])
    df['unit_price'] = parse_money(df['Price'])
    df['qty'] = parse_number(df['Số lượng bán'])
    df['qty_base'] = parse_number(df['SL lẻ'])
    df['unit'] = df['Đơn vị tính']
    df['product_name'] = df['Ten hang']

    # Generate a synthetic order_id based on order_date and row index
    df['order_id'] = df['order_date'].astype(str) + '_' + df.index.astype(str)

    # Identify service fee lines
    df['is_fee'] = df['product_name'].str.contains('Phí phục vụ', case=False, na=False)

    # Create orders table
    orders = df.groupby('order_id').agg({
        'order_datetime': 'first',
        'order_date': 'first',
        'status': 'first',
        'status_raw': 'first',
        'line_amount': 'sum'
    }).reset_index()
    orders.rename(columns={'line_amount': 'goods_total'}, inplace=True)

    # Create order_lines table
    order_lines = df[[
        'order_id', 'product_name', 'unit', 'qty', 'qty_base', 'unit_price', 'line_amount', 'is_fee'
    ]].copy()
    order_lines['line_no'] = order_lines.groupby('order_id').cumcount() + 1

    # Create unit_conversions table
    unit_conversions_data = [
        ('Bia Tiger lon', 'thùng', 24),
        ('Bia Sài Gòn Special lon', 'thùng', 24),
        ('Coca-Cola lon', 'thùng', 24),
        ('Bia Tiger Bạc lon', 'thùng', 24),
        ('Nước suối', 'thùng', 24),
        ('Bia Heineken lon', 'thùng', 24),
        ('Bia Sài Gòn Lager chai', 'két', 20),
        ('7Up lon', 'thùng', 24)
    ]
    unit_conversions = pd.DataFrame(unit_conversions_data, columns=['product_id', 'unit', 'factor'])

    # Create products table
    products = df[['product_name', 'unit']].drop_duplicates().copy()
    products['product_id'] = products['product_name']
    products.rename(columns={'product_name': 'name', 'unit': 'base_unit'}, inplace=True)

    # Prepare metadata
    meta = {
        'industry': 'nhà hàng',
        'has_customer_codes': False,
        'line_no_from_file': False,
        'order_codes_from_file': False,
        'data_end': '2026-06-06',
        'notes': [
            'No order codes in the data; order_id synthesized from date and row index.',
            'Service fee lines identified by product name "Phí phục vụ".',
            'Unit conversions provided in README.'
        ],
        'extra_desc': {}
    }

    # Return the tables and metadata
    tables = {
        'orders': orders,
        'order_lines': order_lines,
        'unit_conversions': unit_conversions,
        'products': products
    }
    return tables, meta
