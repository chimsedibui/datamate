def load(raw):
    # Read the CSV file
    df = raw.read_table('danh_sach_hoa_don.csv', header_rows=1)
    
    # Normalize column names
    df.columns = ['goods_total', 'total_payable', 'order_id', 'customer_name', 
                  'order_date', 'status_raw', 'paid_amount', 'debt_remaining']
    
    # Parse numeric columns
    df['goods_total'] = parse_money(df['goods_total'])
    df['total_payable'] = parse_money(df['total_payable'])
    df['paid_amount'] = parse_money(df['paid_amount'])
    df['debt_remaining'] = parse_money(df['debt_remaining'])
    
    # Parse date column
    df['order_date'] = parse_date(df['order_date'])
    
    # Map status to canonical values
    status_map = {
        'Hoàn thành': 'completed',
        'khách hủy': 'cancelled',
        'Hoàn trả': 'returned',
        'Chờ giao': 'delivering'
    }
    df['status'] = df['status_raw'].map(status_map)
    
    # Handle walk-in customers
    walk_in_labels = ['khách lẻ', 'Khách lẻ', 'kl', 'lẻ']
    df.loc[df['customer_name'].str.strip().str.lower().isin([x.lower() for x in walk_in_labels]), 
           ['customer_name']] = None
    
    # Drop rows with missing required fields
    df = df.dropna(subset=['order_id', 'order_date', 'status'])
    
    # Remove duplicate orders based on order_id
    df = df.drop_duplicates(subset=['order_id'])
    
    # Prepare the orders table
    orders = df[['order_id', 'order_date', 'status', 'status_raw', 'customer_name', 
                 'goods_total', 'total_payable', 'paid_amount', 'debt_remaining']].copy()
    orders['order_datetime'] = orders['order_date']  # No time information available
    orders['branch'] = None
    orders['channel'] = None
    orders['payment_method'] = None
    orders['payment_method_raw'] = None
    orders['order_discount'] = None
    orders['shipping_fee'] = None
    orders['vat'] = None
    orders['note'] = None
    orders['customer_id'] = None  # No customer codes in the data
    orders['employee'] = None
    orders['order_date'] = orders['order_datetime'].dt.date  # Normalize to date
    
    # Prepare metadata
    meta = {
        'industry': 'tạp hoá',
        'has_customer_codes': False,
        'line_no_from_file': False,
        'data_end': '2026-07-03',
        'notes': ['debt recorded in "Chưa trả" column', 'no time information available'],
        'extra_desc': {}
    }
    
    return {'orders': orders}, meta
