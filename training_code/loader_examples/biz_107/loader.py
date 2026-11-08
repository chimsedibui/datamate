def load(raw):
    # Read the raw data from the Excel file
    file = 'SoBanHang.xlsx'
    sheet = 'Sheet1'
    df = raw.read_table(file, sheet=sheet)

    # Rename columns to match canonical schema
    df.rename(columns={
        'Trạng thái xử lý': 'status_raw',
        'Tổng tiền': 'line_amount',
        'ngay_ban': 'order_date',
        'Số lượng': 'qty',
        'Nội dung': 'product_name',
        'don_gia': 'unit_price',
        'Đơn vị (tính)': 'unit',
        'Số lượng lẻ': 'qty_base'
    }, inplace=True)

    # Parse columns
    df['line_amount'] = parse_money(df['line_amount'])
    df['order_date'] = parse_date(df['order_date'], default_year=2026)
    df['unit_price'] = parse_money(df['unit_price'])
    df['qty'] = parse_qty_unit(df['qty'])['qty']
    df['qty_base'] = parse_number(df['qty_base'])

    # Map status
    status_map = {
        'Đã hoàn thành': 'completed',
        'Không giao được': 'cancelled',
        'trả hàng': 'returned',
        'đang gửi': 'delivering'
    }
    df['status'] = df['status_raw'].map(status_map)

    # Generate order_id
    df['order_id'] = (
        df['order_date'].dt.strftime('%Y-%m-%d') + '_' +
        df['status_raw'].fillna('') + '_' +
        df.index.astype(str)
    )

    # Drop rows with null values in required columns
    df.dropna(subset=['order_id', 'qty', 'qty_base'], inplace=True)

    # Create orders table
    orders = df.groupby('order_id').agg({
        'order_date': 'first',
        'status': 'first',
        'status_raw': 'first'
    }).reset_index()
    orders['order_datetime'] = orders['order_date']
    orders['customer_id'] = None
    orders['customer_name'] = None
    orders['customer_group'] = None
    orders['employee'] = None
    orders['branch'] = None
    orders['channel'] = None
    orders['payment_method'] = None
    orders['payment_method_raw'] = None
    orders['goods_total'] = df.groupby('order_id')['line_amount'].sum().values
    orders['order_discount'] = None
    orders['shipping_fee'] = None
    orders['vat'] = None
    orders['total_payable'] = orders['goods_total']
    orders['paid_amount'] = orders['goods_total']
    orders['debt_remaining'] = 0
    orders['note'] = None

    # Create order_lines table
    order_lines = df[['order_id', 'product_name', 'unit', 'qty', 'qty_base', 'unit_price', 'line_amount']].copy()
    order_lines['line_no'] = order_lines.groupby('order_id').cumcount() + 1
    order_lines['product_id'] = order_lines['product_name']
    order_lines['line_discount'] = None
    order_lines['cost_per_unit'] = None
    order_lines['is_fee'] = False

    # Create products table
    products = df[['product_name', 'unit']].drop_duplicates().copy()
    products.rename(columns={'product_name': 'name', 'unit': 'base_unit'}, inplace=True)
    products['product_id'] = products['name']
    products['aliases'] = None
    products['category'] = None
    products['brand'] = None
    products['sell_price'] = None
    products['cost_price'] = None
    products['is_prescription'] = None
    products['active_ingredient'] = None
    products['size'] = None
    products['is_prepared'] = False
    products['active'] = True

    # Create unit_conversions table
    unit_conversions = []
    for product, unit, factor in [
        ('Coca Cola 330ml', 'lốc', 6), ('Coca Cola 330ml', 'thùng', 24),
        ('Pepsi 330ml', 'lốc', 6), ('Pepsi 330ml', 'thùng', 24),
        ('Sting dâu 330ml', 'lốc', 6), ('Sting dâu 330ml', 'thùng', 24),
        ('Red Bull 250ml', 'lốc', 6), ('Red Bull 250ml', 'thùng', 24),
        ('Trà xanh 0 độ 455ml', 'lốc', 6), ('Trà xanh 0 độ 455ml', 'thùng', 24),
        ('Nước suối Aquafina 500ml', 'lốc', 6), ('Nước suối Aquafina 500ml', 'thùng', 24),
        ('Nước suối Lavie 1.5L', 'lốc', 6), ('Nước suối Lavie 1.5L', 'thùng', 12),
        ('Bia Tiger 330ml', 'lốc', 6), ('Bia Tiger 330ml', 'thùng', 24),
        ('Bia Sài Gòn Special 330ml', 'lốc', 6), ('Bia Sài Gòn Special 330ml', 'thùng', 24),
        ('Bia Heineken 330ml', 'lốc', 6), ('Bia Heineken 330ml', 'thùng', 24),
        ('Sữa tươi Vinamilk 180ml', 'lốc', 4), ('Sữa tươi Vinamilk 180ml', 'thùng', 48),
        ('Sữa TH true MILK 180ml', 'lốc', 4), ('Sữa TH true MILK 180ml', 'thùng', 48),
        ('Sữa đặc Ông Thọ 380g', 'thùng', 48),
        ('Sữa chua Vinamilk có đường', 'lốc', 4), ('Sữa chua Vinamilk có đường', 'thùng', 48),
        ('Yakult 65ml', 'lốc', 5), ('Yakult 65ml', 'thùng', 50),
        ('Mì Hảo Hảo tôm chua cay', 'thùng', 30),
        ('Mì Omachi sườn hầm', 'thùng', 30),
        ('Mì 3 Miền tôm hùm', 'thùng', 30),
        ('Phở Vifon bò', 'thùng', 30),
        ('Gạo ST25', 'bao', 5), ('Gạo ST25', 'bao 25kg', 25),
        ('Gạo thơm Jasmine', 'bao', 5), ('Gạo thơm Jasmine', 'bao 25kg', 25),
        ('Đường Biên Hòa 1kg', 'thùng', 12),
        ('Muối I-ốt 500g', 'thùng', 20),
        ('Bột ngọt Ajinomoto 400g', 'thùng', 20),
        ('Hạt nêm Knorr 400g', 'thùng', 20),
        ('Nước mắm Nam Ngư 750ml', 'thùng', 12),
        ('Nước tương Maggi 700ml', 'thùng', 12),
        ('Dầu ăn Tường An 1L', 'thùng', 12),
        ('Dầu ăn Simply 2L', 'thùng', 6),
        ('Tương ớt Chinsu 250g', 'thùng', 24),
        ('Trứng gà so', 'vỉ', 10), ('Trứng gà so', 'hộp 30', 30),
        ('Xúc xích Vissan 5 cây', 'thùng', 20),
        ('Cá hộp 3 Cô Gái 155g', 'lốc', 10),
        ('Đậu phộng rang muối 200g', 'thùng', 20),
        ('Snack Oishi tôm cay 40g', 'thùng', 50),
        ('Snack Lay\'s vị tự nhiên 56g', 'thùng', 32),
        ('Bánh Oreo 133g', 'lốc', 12),
        ('Bánh Chocopie 12 cái', 'thùng', 8),
        ('Bánh quy Cosy 200g', 'thùng', 20),
        ('Kẹo Alpenliebe 100g', 'thùng', 40),
        ('Kẹo cao su Doublemint', 'lốc', 20),
        ('Hạt hướng dương 200g', 'thùng', 20),
        ('Bột giặt OMO 3kg', 'thùng', 4),
        ('Nước giặt Ariel 3.2L', 'thùng', 4),
        ('Nước rửa chén Sunlight 750g', 'thùng', 12),
        ('Nước lau sàn Sunlight 1L', 'thùng', 12),
        ('Kem đánh răng P/S 180g', 'thùng', 36),
        ('Bàn chải Colgate', 'lốc', 12),
        ('Dầu gội Clear 630g', 'thùng', 6),
        ('Sữa tắm Lifebuoy 850g', 'thùng', 6),
        ('Xà bông Lifebuoy 90g', 'lốc', 6), ('Xà bông Lifebuoy 90g', 'thùng', 72),
        ('Giấy vệ sinh Pulppy 10 cuộn', 'thùng', 6),
        ('Khăn giấy Tempo 10 gói', 'thùng', 12),
        ('Băng vệ sinh Diana 8 miếng', 'thùng', 48),
        ('Tã Bobby M 32 miếng', 'thùng', 4),
        ('Thuốc lá Vinataba', 'cây', 10),
        ('Thuốc lá Jet', 'cây', 10),
        ('Pin Panasonic AA', 'hộp', 12),
        ('Nhang Quảng Trí bó', 'thùng', 20)
    ]:
        unit_conversions.append({'product_id': product, 'unit': unit, 'factor': factor})
    unit_conversions = pd.DataFrame(unit_conversions)

    # Prepare meta information
    meta = {
        'industry': 'tạp hoá',
        'has_customer_codes': False,
        'line_no_from_file': False,
        'data_end': '2026-07-22',
        'notes': ['No order IDs or customer names in data', 'Unit conversions provided in README'],
        'extra_desc': {}
    }

    return {
        'orders': orders,
        'order_lines': order_lines,
        'products': products,
        'unit_conversions': unit_conversions
    }, meta
