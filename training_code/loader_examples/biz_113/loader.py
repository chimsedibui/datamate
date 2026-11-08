def load(raw):
    import pandas as pd
    import numpy as np
    import re
    from collections import defaultdict

    # Read the single sheet
    df = raw.read_table('SoBanHang.xlsx', sheet='Sheet1')
    
    # Rename columns for easier handling
    df = df.rename(columns={
        'DonVi': 'unit',
        'Ngày mua': 'order_date_raw',
        'Số lượng lẻ': 'qty_base',
        'thanh_tien': 'line_amount',
        'Đơn giá bán': 'unit_price',
        'SL bán': 'qty',
        'Ghi chú TT': 'status_raw',
        'Mặt hàng': 'product_name'
    })
    
    # Parse date (day-first, year from data)
    df['order_datetime'] = pd.to_datetime(df['order_date_raw'], format='%d/%m/%Y', errors='coerce')
    df['order_date'] = df['order_datetime'].dt.date
    
    # Parse money columns
    df['line_amount'] = df['line_amount'].apply(parse_money)
    df['unit_price'] = df['unit_price'].apply(parse_money)
    df['qty'] = df['qty'].apply(parse_number)
    df['qty_base'] = df['qty_base'].apply(parse_number)
    
    # Status mapping
    status_map = {
        'xong': 'completed',
        'khách hủy': 'cancelled',
        'trả hàng': 'returned',
        'Đang vận chuyển': 'delivering'
    }
    df['status'] = df['status_raw'].map(status_map)
    
    # Since no order code, no customer, no time-of-day: each row is its own order
    # Create stable order_id from row index
    df = df.reset_index(drop=True)
    df['order_id'] = [f'ORD_{i:06d}' for i in range(len(df))]
    df['line_no'] = 1
    
    # Build orders table (one row per order = one row per line)
    orders = pd.DataFrame({
        'order_id': df['order_id'],
        'order_datetime': df['order_datetime'],
        'order_date': df['order_date'],
        'status': df['status'],
        'status_raw': df['status_raw'],
        'customer_id': None,
        'customer_name': None,
        'customer_group': None,
        'employee': None,
        'branch': None,
        'channel': None,
        'payment_method': None,
        'payment_method_raw': None,
        'goods_total': df['line_amount'],
        'order_discount': 0.0,
        'shipping_fee': 0.0,
        'vat': 0.0,
        'total_payable': df['line_amount'],
        'paid_amount': None,
        'debt_remaining': None,
        'note': None
    })
    
    # Build order_lines table
    order_lines = pd.DataFrame({
        'order_id': df['order_id'],
        'line_no': df['line_no'],
        'product_id': df['product_name'],  # no codes, use name as id
        'product_name': df['product_name'],
        'unit': df['unit'],
        'qty': df['qty'],
        'qty_base': df['qty_base'],
        'unit_price': df['unit_price'],
        'line_discount': 0.0,
        'line_amount': df['line_amount'],
        'cost_per_unit': None,
        'is_fee': False
    })
    
    # Build products table (unique products)
    product_info = df.groupby('product_name').agg(
        unit=('unit', 'first'),
        unit_price=('unit_price', 'first')
    ).reset_index()
    
    # Determine base unit and conversion factors from README
    conversion_rules = {
        'Xi măng Hà Tiên PCB40 bao 50kg': {'base_unit': 'bao', 'conversions': {'tấn': 20}},
        'Xi măng INSEE Power-S bao 50kg': {'base_unit': 'bao', 'conversions': {'tấn': 20}},
        'Cát xây tô': {'base_unit': 'm3', 'conversions': {'xe 4m3': 4}},
        'Cát bê tông': {'base_unit': 'm3', 'conversions': {'xe 4m3': 4}},
        'Cát san lấp': {'base_unit': 'm3', 'conversions': {'xe 4m3': 4}},
        'Đá 1x2': {'base_unit': 'm3', 'conversions': {'xe 4m3': 4}},
        'Đá 4x6': {'base_unit': 'm3', 'conversions': {'xe 4m3': 4}},
        'Đá mi bụi': {'base_unit': 'm3', 'conversions': {'xe 4m3': 4}},
        'Gạch ống 8x8x18 Tuynel': {'base_unit': 'viên', 'conversions': {'thiên': 1000}},
        'Gạch thẻ đặc 4x8x18': {'base_unit': 'viên', 'conversions': {'thiên': 1000}},
        'Băng keo điện Nano 20 yard': {'base_unit': 'cuộn', 'conversions': {'cây': 10}},
        'Lưới B40 mạ kẽm khổ 1.5m': {'base_unit': 'm', 'conversions': {'cuộn 10m': 10}},
        'Ngói Đồng Nai 22 viên/m2': {'base_unit': 'viên', 'conversions': {'thiên': 1000}},
        'Bạt xanh cam khổ 4m': {'base_unit': 'm2', 'conversions': {'cuộn 4x50m': 200}},
    }
    
    products_rows = []
    unit_conversions_rows = []
    
    for _, row in product_info.iterrows():
        pname = row['product_name']
        rule = conversion_rules.get(pname, {})
        base_unit = rule.get('base_unit', row['unit'])  # default to the unit seen in data
        
        products_rows.append({
            'product_id': pname,
            'name': pname,
            'aliases': '',
            'category': None,
            'brand': None,
            'base_unit': base_unit,
            'sell_price': row['unit_price'] if row['unit'] == base_unit else None,
            'cost_price': None,
            'is_prescription': False,
            'active_ingredient': None,
            'size': None,
            'is_prepared': False,
            'active': True
        })
        
        # Add conversion factors (larger unit -> base unit)
        for larger_unit, factor in rule.get('conversions', {}).items():
            unit_conversions_rows.append({
                'product_id': pname,
                'unit': larger_unit,
                'factor': float(factor)
            })
        # Base unit always has factor 1
        unit_conversions_rows.append({
            'product_id': pname,
            'unit': base_unit,
            'factor': 1.0
        })
    
    products = pd.DataFrame(products_rows)
    unit_conversions = pd.DataFrame(unit_conversions_rows)
    
    # No customers table (walk-in only)
    # No payments, purchases, stock, etc. recorded
    
    tables = {
        'orders': orders,
        'order_lines': order_lines,
        'products': products,
        'unit_conversions': unit_conversions
    }
    
    meta = {
        'industry': 'vật liệu xây dựng',
        'has_customer_codes': False,
        'line_no_from_file': False,
        'data_end': '2026-08-09',
        'notes': [
            'No order codes or customer names in source; each row treated as a separate order',
            'Qty_base column already converted to base unit per README conversion rules',
            'Status values: xong=completed, khách hủy=cancelled, trả hàng=returned, Đang vận chuyển=delivering',
            'Unit conversions defined in README for specific products; other products use shown unit as base unit',
            'No payment, cost, or inventory data recorded'
        ],
        'extra_desc': {}
    }
    
    return tables, meta
