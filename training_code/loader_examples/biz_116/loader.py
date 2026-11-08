def load(raw):
    # Read the sheets from the Excel file
    sales_details = raw.read_table("Bao_cao_ban_hang.xlsx", sheet="So_chi_tiet_ban_hang")
    product_list = raw.read_table("Bao_cao_ban_hang.xlsx", sheet="Danh sách hàng hóa")
    delivery_notes = raw.read_table("Bao_cao_ban_hang.xlsx", sheet="Bảng kê phiếu giao hàng")
    debt_details = raw.read_table("Bao_cao_ban_hang.xlsx", sheet="Chi tiết công nợ theo hạn")

    # Parse sales details
    sales_details['Ngày'] = parse_datetime(sales_details['Ngày'])
    sales_details['order_date'] = parse_date(sales_details['Ngày'])
    sales_details['order_id'] = sales_details['Mã đơn hàng']
    sales_details['order_datetime'] = sales_details['Ngày']
    sales_details['status_raw'] = sales_details['Trạng thái']
    sales_details['status'] = sales_details['Trạng thái'].str.lower().map({
        'ok': 'completed',
        'thành công': 'completed',
        'khách hủy': 'cancelled',
        'không giao được': 'cancelled',
        'trả hàng': 'returned',
        'đã trả hàng': 'returned',
        'đang vận chuyển': 'delivering'
    })
    sales_details['customer_name'] = sales_details['Khách hàng'].replace({'Khách lẻ': None})
    sales_details['customer_id'] = None
    sales_details['goods_total'] = parse_money(sales_details['Thành tiền'])

    # Parse the 'Số lượng' column to extract numeric values
    def parse_quantity(s):
        match = re.search(r'(\d+(\.\d+)?)', str(s))
        return float(match.group(1)) if match else None

    sales_details['qty'] = sales_details['Số lượng'].apply(parse_quantity)
    sales_details['unit'] = sales_details['Đơn vị tính']
    sales_details['line_amount'] = parse_money(sales_details['Thành tiền'])
    sales_details['qty_base'] = parse_number(sales_details['Số lượng quy đổi'])
    sales_details['product_name'] = sales_details['Tên hàng hóa']
    sales_details['unit_price'] = parse_money(sales_details['Đơn giá'])
    sales_details['line_discount'] = 0.0  # No line discount column in the data
    sales_details['is_fee'] = sales_details['Tên hàng hóa'].str.contains('phí phục vụ', case=False, na=False)

    # Group by order_id to forward fill order-level data
    ffill_group(sales_details, 'order_id', ['order_datetime', 'order_date', 'status', 'status_raw', 'customer_name'])

    # Create orders table
    orders = sales_details.groupby('order_id').agg({
        'order_datetime': 'first',
        'order_date': 'first',
        'status': 'first',
        'status_raw': 'first',
        'customer_name': 'first',
        'goods_total': 'sum'
    }).reset_index()

    # Create order_lines table
    sales_details['line_no'] = sales_details.groupby('order_id').cumcount() + 1
    order_lines = sales_details[[
        'order_id', 'line_no', 'product_name', 'unit', 'qty', 'qty_base', 'unit_price', 'line_discount', 'line_amount', 'is_fee'
    ]]

    # Parse product list
    product_list['product_id'] = product_list['Mã hàng hóa']
    product_list['name'] = product_list['Tên hàng hóa']
    product_list['category'] = product_list['Nhóm hàng hóa']
    product_list['base_unit'] = product_list['Đơn vị tính']
    product_list['cost_price'] = parse_money(product_list['Giá vốn'])
    product_list['active'] = True  # Assuming all products are active
    products = product_list[['product_id', 'name', 'category', 'base_unit', 'cost_price', 'active']]

    # Parse delivery notes
    delivery_notes['delivery_date'] = parse_date(delivery_notes['Ngày giao hàng'])
    deliveries = delivery_notes.rename(columns={
        'Mã đơn hàng': 'order_id',
        'Lần giao': 'delivery_id',
        'Tên hàng hóa': 'product_name',
        'Số lượng giao': 'qty_base',
        'Biển số xe': 'note',
        'Người giao hàng': 'status'
    })[['delivery_id', 'order_id', 'delivery_date', 'status', 'note']]

    # Parse debt details
    debt_details['due_date'] = parse_date(debt_details['Hạn thanh toán'])
    debt_details = debt_details.rename(columns={
        'Mã đơn hàng': 'order_id',
        'Số ngày được nợ': 'note'
    })

    # Create unit_conversions table
    conversion_data = [
        ('SP0001', 'tấn', 20),
        ('SP0002', 'tấn', 20),
        ('SP0003', 'xe 4m3', 4),
        ('SP0004', 'xe 4m3', 4),
        ('SP0005', 'xe 4m3', 4),
        ('SP0006', 'xe 4m3', 4),
        ('SP0007', 'xe 4m3', 4),
        ('SP0008', 'xe 4m3', 4),
        ('SP0009', 'thiên', 1000),
        ('SP0010', 'thiên', 1000),
        ('SP0011', 'cây', 10),
        ('SP0012', 'cuộn 10m', 10),
        ('SP0013', 'thiên', 1000),
        ('SP0014', 'cuộn 4x50m', 200)
    ]
    unit_conversions = pd.DataFrame(conversion_data, columns=['product_id', 'unit', 'factor'])

    # Meta information
    meta = {
        'industry': 'vật liệu xây dựng',
        'has_customer_codes': False,
        'line_no_from_file': False,
        'data_end': '2026-09-03',
        'notes': [
            'Units are recorded in various forms and need conversion to base units',
            'Debt is recorded with due dates and allowed days',
            'Delivery notes are recorded for certain product categories'
        ],
        'extra_desc': {},
        'dropped_duplicate_lines': 0
    }

    return {
        'orders': orders,
        'order_lines': order_lines,
        'products': products,
        'unit_conversions': unit_conversions,
        'deliveries': deliveries
    }, meta
