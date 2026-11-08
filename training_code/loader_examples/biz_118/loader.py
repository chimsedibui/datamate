def load(raw):
    import pandas as pd
    import numpy as np
    from datetime import datetime

    # Helper function to parse money with 'tr' (triệu) and commas
    def parse_money(s):
        if pd.isna(s):
            return np.nan
        s = str(s).strip()
        if 'tr' in s:
            s = s.replace('tr', '000000').replace(',', '')
        elif 'triệu' in s:
            s = s.replace('triệu', '000000').replace(',', '')
        elif 'đ' in s:
            s = s.replace('đ', '')
        elif 'đồng' in s:
            s = s.replace('đồng', '')
        elif 'VND' in s:
            s = s.replace('VND', '')
        try:
            return float(s.replace(',', ''))
        except:
            return np.nan

    # Helper function to parse quantity from strings like '1 cái'
    def parse_quantity(s):
        if pd.isna(s):
            return np.nan
        s = str(s).strip()
        try:
            # Extract numeric part using regex
            import re
            match = re.search(r'[-+]?\d*\.?\d+|\d+', s)
            if match:
                return float(match.group())
            return np.nan
        except:
            return np.nan

    # Read sales data
    sales_files = raw.list_files()
    sales_dfs = []
    for file in sales_files:
        if file.startswith("So_chi_tiet_ban_hang.xlsx"):
            sheets = raw.sheet_names(file)
            for sheet in sheets:
                df = raw.read_table(file, sheet=sheet, header_rows=1)
                sales_dfs.append(df)
    sales_df = pd.concat(sales_dfs, ignore_index=True)

    # Process products
    product_df = raw.read_table("Danh sách hàng hóa.xlsx", sheet="Sheet1", header_rows=1)
    product_name_to_id = dict(zip(product_df['Tên hàng hóa'], product_df['Mã hàng hóa']))
    products = []
    for _, row in product_df.iterrows():
        product_id = row['Mã hàng hóa']
        name = row['Tên hàng hóa']
        category = row['Nhóm hàng hóa']
        base_unit = row['Đơn vị tính']
        products.append({
            'product_id': product_id,
            'name': name,
            'aliases': None,
            'category': category,
            'brand': None,
            'base_unit': base_unit,
            'sell_price': 0,
            'cost_price': 0,
            'is_prescription': False,
            'active_ingredient': None,
            'size': None,
            'is_prepared': False,
            'active': True
        })
    products_df = pd.DataFrame(products)

    # Process unit conversions from README
    unit_conversions = []
    # Xi măng
    for product_name in ['Xi măng Hà Tiên PCB40 bao 50kg', 'Xi măng INSEE Power-S bao 50kg']:
        if product_name in product_name_to_id:
            product_id = product_name_to_id[product_name]
            unit_conversions.append({
                'product_id': product_id,
                'unit': 'tấn',
                'factor': 20
            })
    # Cát
    for product_name in ['Cát xây tô', 'Cát bê tông', 'Cát san lấp']:
        if product_name in product_name_to_id:
            product_id = product_name_to_id[product_name]
            unit_conversions.append({
                'product_id': product_id,
                'unit': 'xe 4m3',
                'factor': 4
            })
    # Gạch
    for product_name in ['Gạch ống 8x8x18 Tuynel', 'Gạch thẻ đặc 4x8x18']:
        if product_name in product_name_to_id:
            product_id = product_name_to_id[product_name]
            unit_conversions.append({
                'product_id': product_id,
                'unit': 'thiên',
                'factor': 1000
            })
    # Băng keo
    if 'Băng keo điện Nano 20 yard' in product_name_to_id:
        product_id = product_name_to_id['Băng keo điện Nano 20 yard']
        unit_conversions.append({
            'product_id': product_id,
            'unit': 'cây',
            'factor': 10
        })
    # Lưới
    if 'Lưới B40 mạ kẽm khổ 1.5m' in product_name_to_id:
        product_id = product_name_to_id['Lưới B40 mạ kẽm khổ 1.5m']
        unit_conversions.append({
            'product_id': product_id,
            'unit': 'cuộn 10m',
            'factor': 10
        })
    # Ngói
    if 'Ngói Đồng Nai 22 viên/m2' in product_name_to_id:
        product_id = product_name_to_id['Ngói Đồng Nai 22 viên/m2']
        unit_conversions.append({
            'product_id': product_id,
            'unit': 'thiên',
            'factor': 1000
        })
    # Bạt
    if 'Bạt xanh cam khổ 4m' in product_name_to_id:
        product_id = product_name_to_id['Bạt xanh cam khổ 4m']
        unit_conversions.append({
            'product_id': product_id,
            'unit': 'cuộn 4x50m',
            'factor': 200
        })
    unit_conversions_df = pd.DataFrame(unit_conversions)

    # Process orders and order_lines
    orders = []
    order_lines = []
    for _, row in sales_df.iterrows():
        order_id = row['Mã đơn hàng']
        date_str = row['Ngày']
        order_date = pd.to_datetime(date_str, dayfirst=True)
        order_datetime = order_date.replace(hour=0, minute=0, second=0)
        status_raw = row['Trạng thái']
        status = map_status(status_raw, {
            'hủy': 'cancelled',
            'huỷ': 'cancelled',
            'đã hủy đơn': 'cancelled',
            'đã hủy': 'cancelled',  # Added to cover more variations
            'trả hàng': 'returned',
            'hàng trả về': 'returned',
            'đổi trả': 'returned',
            'hoàn': 'returned',
            'đã xong': 'completed',
            'hoàn thành': 'completed',
            'ok': 'completed',
            'đã lấy': 'completed',
            'đang giao': 'delivering',
            'chờ giao': 'delivering'
        })
        if pd.isna(status):
            # If status is still NaN, default to 'completed'
            status = 'completed'

        customer_name = row['Khách hàng']
        customer_id = None
        if customer_name in ['Khách lẻ', 'khách vãng lai', 'lẻ']:
            customer_name = None
        else:
            customer_id = customer_name  # Assuming customer_id is the name if no code exists

        # Map product info
        product_name = row['Tên hàng hóa']
        product_id = product_name_to_id.get(product_name, product_name)
        unit = row['Đơn vị tính']
        qty_str = row['Số lượng']
        qty = parse_quantity(qty_str)
        qty_base_str = row['Số lượng quy đổi']
        qty_base = parse_quantity(qty_base_str)
        unit_price_str = row['Đơn giá']
        unit_price = parse_money(unit_price_str)
        line_amount_str = row['Thành tiền']
        line_amount = parse_money(line_amount_str)

        # Apply unit conversions
        if product_id in product_name_to_id.values():
            # Find matching product in products_df
            product_row = products_df[products_df['product_id'] == product_id].iloc[0]
            base_unit = product_row['base_unit']
            if unit != base_unit:
                # Apply conversion from README
                if unit == 'tấn' and base_unit == 'bao':
                    factor = 20
                elif unit == 'xe 4m3' and base_unit == 'm3':
                    factor = 4
                elif unit == 'thiên' and base_unit == 'viên':
                    factor = 1000
                elif unit == 'cây' and base_unit == 'cuộn':
                    factor = 10
                elif unit == 'cuộn 10m' and base_unit == 'm':
                    factor = 10
                elif unit == 'thiên' and base_unit == 'viên':
                    factor = 1000
                elif unit == 'cuộn 4x50m' and base_unit == 'm2':
                    factor = 200
                else:
                    factor = 1
                qty_base = qty * factor
        else:
            qty_base = qty  # Default to same unit if no conversion found

        # Add to order_lines
        order_lines.append({
            'order_id': order_id,
            'line_no': len(order_lines) + 1,
            'product_id': product_id,
            'product_name': product_name,
            'unit': unit,
            'qty': qty,
            'qty_base': qty_base,
            'unit_price': unit_price,
            'line_discount': 0,
            'line_amount': line_amount,
            'cost_per_unit': 0,
            'is_fee': False
        })

        # Aggregate orders
        orders.append({
            'order_id': order_id,
            'order_datetime': order_datetime,
            'order_date': order_date,
            'status': status,
            'status_raw': status_raw,
            'customer_id': customer_id,
            'customer_name': customer_name,
            'customer_group': None,
            'employee': None,
            'branch': None,
            'channel': None,
            'payment_method': None,
            'payment_method_raw': None,
            'goods_total': line_amount,
            'order_discount': 0,
            'shipping_fee': 0,
            'vat': 0,
            'total_payable': line_amount,
            'paid_amount': 0,
            'debt_remaining': 0,
            'note': None
        })

    # Deduplicate orders by order_id
    orders_df = pd.DataFrame(orders).groupby('order_id').agg({
        'order_datetime': 'first',
        'order_date': 'first',
        'status': 'first',
        'status_raw': 'first',
        'customer_id': 'first',
        'customer_name': 'first',
        'customer_group': 'first',
        'employee': 'first',
        'branch': 'first',
        'channel': 'first',
        'payment_method': 'first',
        'payment_method_raw': 'first',
        'goods_total': 'sum',
        'order_discount': 'sum',
        'shipping_fee': 'sum',
        'vat': 'sum',
        'total_payable': 'sum',
        'paid_amount': 'sum',
        'debt_remaining': 'sum',
        'note': 'first'
    }).reset_index()

    # Process deliveries
    deliveries_df = raw.read_table("Bảng kê phiếu giao hàng.xlsx", sheet="Sheet1", header_rows=1)
    deliveries = []
    for _, row in deliveries_df.iterrows():
        delivery_id = f"{row['Mã đơn hàng']}_{row['Lần giao']}"
        delivery_date = pd.to_datetime(row['Ngày giao hàng'], dayfirst=True)
        deliveries.append({
            'delivery_id': delivery_id,
            'order_id': row['Mã đơn hàng'],
            'delivery_date': delivery_date,
            'status': 'delivered',
            'note': None
        })
    deliveries_df = pd.DataFrame(deliveries)

    # Meta information
    meta = {
        'industry': 'vật liệu xây dựng',
        'has_customer_codes': False,
        'line_no_from_file': False,
        'data_end': '2026-09-10',
        'notes': [
            'Số lượng quy đổi đã được tính theo đơn vị cơ bản',
            'Debt_remaining is 0 as no data provided',
            'Unit conversions based on README rules',
            'Product ID mapping from product name to Mã hàng hóa',
            'Status mapped from status_raw with expanded dictionary and default to "completed"',
            'Money parsing fixed for "tr" (triệu) and commas',
            'Quantity parsing fixed for strings like "1 cái"'
        ],
        'extra_desc': {}
    }

    tables = {
        'orders': orders_df,
        'order_lines': pd.DataFrame(order_lines),
        'products': products_df,
        'unit_conversions': unit_conversions_df,
        'deliveries': deliveries_df
    }

    return tables, meta
