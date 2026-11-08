def load(raw):
    import pandas as pd
    import re

    # Read product catalog
    products_df = raw.read_table('Danh sách hàng hóa.xlsx', sheet='Sheet1', header_rows=1)
    products_df = products_df.rename(columns={
        'Mã hàng hóa': 'product_id',
        'Tên hàng hóa': 'name',
        'Nhóm hàng hóa': 'category',
        'Đơn vị tính': 'base_unit',
        'Giá bán': 'sell_price',
        'Giá vốn': 'cost_price',
    })
    products_df['sell_price'] = parse_money(products_df['sell_price'])
    products_df['cost_price'] = parse_money(products_df['cost_price'])
    products_df = products_df[['product_id', 'name', 'category', 'base_unit', 'sell_price', 'cost_price']]

    # Read sales data
    sales_files = ['So_chi_tiet_ban_hang_1.xlsx', 'So_chi_tiet_ban_hang_2026.xlsx']
    sales_dfs = []
    for file in sales_files:
        sheets = raw.sheet_names(file)
        for sheet in sheets:
            df = raw.read_table(file, sheet=sheet, header_rows=1)
            sales_dfs.append(df)
    sales_df = pd.concat(sales_dfs, ignore_index=True)

    # Parse columns
    sales_df['Ngày'] = parse_datetime(sales_df['Ngày'], default_year=None)
    sales_df['Đơn giá'] = parse_money(sales_df['Đơn giá'])
    sales_df['Thành tiền'] = parse_money(sales_df['Thành tiền'])
    
    # Fix: Convert 'Số lượng' to string first to avoid TypeError
    sales_df['Số lượng'] = sales_df['Số lượng'].astype(str).apply(lambda x: float(re.search(r'\d+', x).group()))
    sales_df['Số lượng quy đổi'] = sales_df['Số lượng quy đổi'].astype(float)

    # Process orders and order_lines
    orders = []
    order_lines = []
    for _, group in sales_df.groupby('Mã đơn hàng'):
        order_id = group['Mã đơn hàng'].iloc[0]
        status_raw = group['Trạng thái'].iloc[0]
        status = map_status(status_raw, {
            'xong': 'completed',
            'đã hoàn thành': 'completed',
            'cancel': 'cancelled',
            'Đã hủy đơn': 'cancelled',
            'trả hàng': 'returned',
            'Đang vận chuyển': 'delivering'
        })
        order_date = group['Ngày'].iloc[0].date()
        order_datetime = group['Ngày'].iloc[0]
        orders.append({
            'order_id': order_id,
            'order_datetime': order_datetime,
            'order_date': order_date,
            'status': status,
            'status_raw': status_raw,
            'customer_id': None,
            'customer_name': None,
            'customer_group': None,
            'employee': None,
            'branch': None,
            'channel': None,
            'payment_method': None,
            'payment_method_raw': None,
            'goods_total': group['Thành tiền'].sum(),
            'order_discount': 0,
            'shipping_fee': 0,
            'vat': 0,
            'total_payable': 0,
            'paid_amount': 0,
            'debt_remaining': 0,
            'note': None,
        })

        for idx, row in group.iterrows():
            line_no = idx + 1
            product_name = row['Tên hàng hóa']
            # Match product_id from product name using normalized names
            product_name_normalized = norm_text(product_name)
            product_id = products_df[products_df['name'].apply(norm_text) == product_name_normalized]['product_id'].values[0] if not products_df[products_df['name'].apply(norm_text) == product_name_normalized].empty else product_name
            unit = row['Đơn vị tính']
            qty = row['Số lượng']
            qty_base = row['Số lượng quy đổi']
            unit_price = row['Đơn giá']
            line_amount = row['Thành tiền']
            line_discount = unit_price * qty - line_amount
            # Match cost_price from product_id
            cost_per_unit = products_df[products_df['product_id'] == product_id]['cost_price'].values[0] if not products_df[products_df['product_id'] == product_id].empty else 0
            is_fee = False
            order_lines.append({
                'order_id': order_id,
                'line_no': line_no,
                'product_id': product_id,
                'product_name': product_name,
                'unit': unit,
                'qty': qty,
                'qty_base': qty_base,
                'unit_price': unit_price,
                'line_discount': line_discount,
                'line_amount': line_amount,
                'cost_per_unit': cost_per_unit,
                'is_fee': is_fee,
            })

    # Build tables
    tables = {
        'orders': pd.DataFrame(orders),
        'order_lines': pd.DataFrame(order_lines),
        'products': products_df,
    }

    # Meta
    meta = {
        'industry': 'cà phê',
        'has_customer_codes': False,
        'line_no_from_file': False,
        'data_end': '2026-07-31',
        'notes': [
            'Quy đổi đơn vị: 1 hộp = 6 cái for certain products',
            'Số lượng quy đổi is already provided in the sales data',
            'Product names are normalized for matching with product catalog'
        ],
        'extra_desc': {},
    }

    return tables, meta
