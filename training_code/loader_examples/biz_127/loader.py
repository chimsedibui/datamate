def load(raw):
    import pandas as pd
    import numpy as np

    # ---------- Read raw files ----------
    orders_raw = raw.read_table('don_hang.csv')
    lines_raw = raw.read_table('chi_tiet_don.csv')
    products_raw = raw.read_table('san_pham.csv')
    customers_raw = raw.read_table('khach_hang.csv')
    payments_raw = raw.read_table('thanh_toan.csv')

    # ---------- Parse orders ----------
    orders = pd.DataFrame()
    orders['order_id'] = orders_raw['Mã chứng từ'].astype(str).str.strip()
    orders['order_datetime'] = parse_datetime(orders_raw['Ngày'])
    orders['order_date'] = orders['order_datetime'].dt.date
    # status mapping
    status_map = {
        'completed': 'completed',
        'cancelled': 'cancelled',
        'returned': 'returned',
        'shipping': 'delivering',
    }
    orders['status_raw'] = orders_raw['Tình trạng'].astype(str).str.strip().str.lower()
    orders['status'] = orders['status_raw'].map(status_map)
    # customer
    orders['customer_id'] = orders_raw['Customer ID'].astype(str).str.strip()
    orders['customer_id'] = orders['customer_id'].replace({'': None, 'nan': None, 'None': None})
    # channel
    channel_map = {'store': 'tại quầy', 'zalo_oa': 'zalo', 'phone': 'điện thoại'}
    orders['channel'] = orders_raw['Loại bán'].astype(str).str.strip().map(channel_map).fillna(orders_raw['Loại bán'])
    # payment method
    pm_map = {'cash': 'cash', 'transfer': 'transfer', 'ewallet': 'ewallet', 'cod': 'cod', 'debt': 'debt'}
    orders['payment_method_raw'] = orders_raw['Trả bằng'].astype(str).str.strip().str.lower()
    orders['payment_method'] = orders['payment_method_raw'].map(pm_map).fillna('other')
    # amounts
    orders['total_payable'] = parse_money(orders_raw['Tổng đơn'])
    orders['paid_amount'] = parse_money(orders_raw['Khách trả'])
    orders['debt_remaining'] = parse_money(orders_raw['Còn nợ'])
    orders['note'] = orders_raw['Diễn giải'].astype(str).str.strip().replace({'': None, 'nan': None})
    # employee, branch not available
    orders['employee'] = None
    orders['branch'] = None
    orders['customer_name'] = None  # will join later
    orders['customer_group'] = None
    orders['goods_total'] = 0.0  # will compute from lines
    orders['order_discount'] = 0.0
    orders['shipping_fee'] = 0.0
    orders['vat'] = 0.0

    # ---------- Parse order_lines ----------
    lines = pd.DataFrame()
    lines['order_id'] = lines_raw['Mã đơn hàng'].astype(str).str.strip()
    lines['line_no'] = lines_raw['No.'].astype(int)
    lines['product_id'] = lines_raw['Mã mặt hàng'].astype(str).str.strip()
    lines['product_name'] = None  # will join from products
    lines['unit'] = lines_raw['DVT'].astype(str).str.strip()
    lines['qty'] = parse_number(lines_raw['SoLuong'])
    lines['qty_base'] = parse_number(lines_raw['Số lượng lẻ'])
    lines['unit_price'] = parse_money(lines_raw['Đơn giá (đ)'])
    lines['line_amount'] = parse_money(lines_raw['Amount'])
    lines['line_discount'] = 0.0  # not separately recorded; line_amount is after line discount
    lines['cost_per_unit'] = None  # not in sales lines
    lines['is_fee'] = False  # will mark for "Phí phục vụ"

    # ---------- Parse products ----------
    products = pd.DataFrame()
    products['product_id'] = products_raw['Mã sản phẩm bán'].astype(str).str.strip()
    products['name'] = products_raw['Tên SP'].astype(str).str.strip()
    products['category'] = products_raw['Danh mục'].astype(str).str.strip()
    products['base_unit'] = products_raw['Đơn vị cơ bản'].astype(str).str.strip()
    products['sell_price'] = parse_money(products_raw['Giá niêm yết'])
    products['cost_price'] = None
    products['brand'] = products_raw['Thương hiệu'].astype(str).str.strip().replace({'': None, 'nan': None})
    products['aliases'] = None
    products['is_prescription'] = False
    products['active_ingredient'] = None
    products['size'] = None
    products['is_prepared'] = products['category'].isin(['chiên xào', 'lẩu', 'gỏi', 'rau', 'nướng', 'cơm mì cháo', 'hải sản', 'khai vị'])  # made on site
    products['active'] = products_raw['Được bán trực tiếp'].astype(str).str.strip() == '1'

    # ---------- Join product_name into lines ----------
    prod_name_map = products.set_index('product_id')['name'].to_dict()
    lines['product_name'] = lines['product_id'].map(prod_name_map)

    # ---------- Mark "Phí phục vụ" lines ----------
    fee_product_ids = products.loc[products['category'] == 'phí dịch vụ', 'product_id'].tolist()
    lines.loc[lines['product_id'].isin(fee_product_ids), 'is_fee'] = True

    # ---------- Compute goods_total per order from lines ----------
    goods_total = lines.groupby('order_id')['line_amount'].sum().rename('goods_total')
    orders = orders.set_index('order_id')
    orders['goods_total'] = goods_total
    orders = orders.reset_index()
    # order_discount = goods_total - total_payable (if positive)
    orders['order_discount'] = (orders['goods_total'] - orders['total_payable']).clip(lower=0)

    # ---------- Join customer_name into orders ----------
    cust_name_map = customers_raw.set_index('Mã KH')['Khách hàng (tên)'].to_dict()
    orders['customer_name'] = orders['customer_id'].map(cust_name_map)
    # customer_group from "Loại hình khách"
    cust_group_map = customers_raw.set_index('Mã KH')['Loại hình khách'].to_dict()
    orders['customer_group'] = orders['customer_id'].map(cust_group_map)

    # ---------- Parse customers ----------
    customers = pd.DataFrame()
    customers['customer_id'] = customers_raw['Mã KH'].astype(str).str.strip()
    customers['name'] = customers_raw['Khách hàng (tên)'].astype(str).str.strip()
    customers['group'] = customers_raw['Loại hình khách'].astype(str).str.strip()
    customers['phone'] = customers_raw['Di động'].astype(str).str.strip()
    # registered_date not available
    customers['registered_date'] = pd.NaT

    # ---------- Parse payments ----------
    payments = pd.DataFrame()
    payments['payment_id'] = payments_raw['Mã GD'].astype(str).str.strip()
    payments['order_id'] = payments_raw['Số CT'].astype(str).str.strip()
    payments['paid_datetime'] = parse_datetime(payments_raw['Ngày thu'])
    payments['amount'] = parse_money(payments_raw['Giá trị'])
    pm_map_pay = {'cash': 'cash', 'transfer': 'transfer', 'ewallet': 'ewallet', 'cod': 'cod'}
    payments['method'] = payments_raw['HTTT'].astype(str).str.strip().str.lower().map(pm_map_pay).fillna('other')
    payments['kind'] = 'payment'

    # ---------- Unit conversions from README ----------
    # Build unit_conversions table
    conv_rows = []
    # Map product names to product_ids for the conversion rules
    name_to_id = products.set_index('name')['product_id'].to_dict()
    # Helper to add conversion if product exists
    def add_conv(product_name, larger_unit, factor):
        pid = name_to_id.get(product_name)
        if pid:
            conv_rows.append({'product_id': pid, 'unit': larger_unit, 'factor': factor})
    add_conv('Bia Tiger lon', 'thùng', 24)
    add_conv('Bia Sài Gòn Special lon', 'thùng', 24)
    add_conv('Coca-Cola lon', 'thùng', 24)
    add_conv('Bia Tiger Bạc lon', 'thùng', 24)
    add_conv('Nước suối', 'thùng', 24)
    add_conv('Bia Heineken lon', 'thùng', 24)
    add_conv('Bia Sài Gòn Lager chai', 'két', 20)
    add_conv('7Up lon', 'thùng', 24)
    # Also add base unit factor 1 for each product
    for _, prow in products.iterrows():
        conv_rows.append({'product_id': prow['product_id'], 'unit': prow['base_unit'], 'factor': 1.0})
    unit_conversions = pd.DataFrame(conv_rows).drop_duplicates(subset=['product_id', 'unit'])

    # ---------- Build tables dict ----------
    tables = {
        'orders': orders,
        'order_lines': lines,
        'products': products,
        'customers': customers,
        'payments': payments,
        'unit_conversions': unit_conversions,
    }

    # ---------- Meta ----------
    meta = {
        'industry': 'nhà hàng',
        'has_customer_codes': True,
        'line_no_from_file': True,
        'data_end': '2026-07-28',
        'notes': [
            'Tổng đơn = total_payable (số khách phải trả cả đơn). goods_total = sum line_amount. order_discount = goods_total - total_payable (nếu >0).',
            'Khách trả = paid_amount trên đơn. Còn nợ = debt_remaining.',
            'Trả bằng trên đơn: cash, transfer, ewallet, cod, debt. Thanh toán sau (thanh_toan.csv) ghi từng lần trả, method tương tự.',
            'Phí phục vụ (phí phòng lạnh) là dòng hàng category="phí dịch vụ", is_fee=True, vẫn tính doanh thu.',
            'Số lượng lẻ trong chi_tiet_don đã là qty_base (quy về đơn vị cơ bản).',
            'Quy đổi đơn vị theo README: các loại bia/nước 1 thùng=24 lon/chai, Bia Sài Gòn Lager chai 1 két=20 chai.',
            'Khách lẻ: Customer ID trống -> customer_id=null, customer_name=null.',
            'Loại bán: store=tại quầy, zalo_oa=zalo, phone=điện thoại.',
            'Tình trạng: completed, cancelled, returned, shipping(=delivering).',
            'Dữ liệu đến 28/07/2026.',
        ],
        'extra_desc': {},
    }

    return tables, meta
