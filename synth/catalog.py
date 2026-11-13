"""Product catalogues of common Vietnamese small businesses, written from general knowledge (not from any test set).

Each item is (name, base unit, base-unit price in VND, [(bigger unit, factor), ...]). A "service" industry sells work as
well as parts; F&B industries sell made-to-order items (no stock). Prices are typical retail prices and only need to be
plausible.
"""

INDUSTRIES = {
    "tap_hoa": {"label": "Tạp hoá", "kind": "retail", "orders": (40, 140), "groups": {
        "Đồ uống": [("Coca Cola 330ml", "lon", 10000, [("lốc", 6), ("thùng", 24)]), ("Nước suối Lavie 500ml", "chai", 5000, [("thùng", 24)]),
                    ("Trà xanh 0 độ 455ml", "chai", 10000, [("thùng", 24)]), ("Bia Sài Gòn lager 330ml", "lon", 14000, [("thùng", 24)])],
        "Mì gói": [("Mì Hảo Hảo tôm chua cay", "gói", 4500, [("thùng", 30)]), ("Mì Omachi sườn hầm", "gói", 8000, [("thùng", 30)])],
        "Gia vị": [("Nước mắm Nam Ngư 500ml", "chai", 32000, [("thùng", 12)]), ("Đường Biên Hoà 1kg", "gói", 28000, []),
                   ("Dầu ăn Simply 1L", "chai", 58000, [("thùng", 12)]), ("Bột ngọt Ajinomoto 400g", "gói", 32000, [])],
        "Gạo": [("Gạo ST25", "kg", 32000, [("bao", 5)]), ("Gạo tám thơm", "kg", 22000, [("bao", 10)])],
        "Sữa": [("Sữa tươi Vinamilk 180ml", "hộp", 8000, [("lốc", 4), ("thùng", 48)]), ("Sữa đặc Ông Thọ 380g", "hộp", 27000, [])]}},
    "ca_phe": {"label": "Cà phê", "kind": "fnb", "orders": (60, 180), "groups": {
        "Cà phê": [("Cà phê đen (M)", "ly", 25000, []), ("Cà phê sữa (M)", "ly", 29000, []), ("Bạc xỉu (M)", "ly", 32000, []),
                   ("Cà phê sữa (L)", "ly", 35000, []), ("Americano (M)", "ly", 39000, [])],
        "Trà": [("Trà đào cam sả (M)", "ly", 45000, []), ("Trà vải (M)", "ly", 45000, []), ("Trà tắc (L)", "ly", 30000, [])],
        "Đá xay": [("Cookie đá xay (L)", "ly", 55000, []), ("Matcha đá xay (L)", "ly", 55000, [])],
        "Bánh": [("Bánh mì que", "cái", 15000, []), ("Croissant bơ", "cái", 35000, [])]}},
    "nha_thuoc": {"label": "Nhà thuốc", "kind": "retail", "orders": (30, 110), "groups": {
        "Giảm đau hạ sốt": [("Paracetamol 500mg", "viên", 1000, [("vỉ", 10), ("hộp", 100)]), ("Efferalgan 500mg", "viên", 3500, [("hộp", 16)])],
        "Kháng sinh": [("Amoxicillin 500mg", "viên", 2000, [("vỉ", 10), ("hộp", 100)]), ("Cefuroxim 500mg", "viên", 9000, [("hộp", 10)])],
        "Vitamin": [("Vitamin C 500mg", "viên", 1500, [("lọ", 100)]), ("Berocca", "viên", 8000, [("tuýp", 10)])],
        "Dụng cụ y tế": [("Khẩu trang y tế", "cái", 1000, [("hộp", 50)]), ("Nhiệt kế điện tử", "cái", 85000, []), ("Bông y tế 100g", "gói", 25000, [])]}},
    "vlxd": {"label": "Vật liệu xây dựng", "kind": "retail", "orders": (8, 40), "groups": {
        "Xi măng": [("Xi măng Hà Tiên PCB40", "bao", 92000, [("tấn", 20)]), ("Xi măng Nghi Sơn PCB40", "bao", 85000, [("tấn", 20)])],
        "Sắt thép": [("Thép cuộn phi 6", "kg", 16000, [("tấn", 1000)]), ("Thép thanh D10", "cây", 98000, [])],
        "Gạch": [("Gạch ống 8x18", "viên", 1300, [("thiên", 1000)]), ("Gạch men 60x60", "viên", 52000, [("thùng", 4)])],
        "Cát đá": [("Cát xây", "khối", 330000, []), ("Đá 1x2", "khối", 380000, [])],
        "Sơn": [("Sơn Dulux nội thất 5L", "thùng", 820000, []), ("Bột trét Joton", "bao", 230000, [])]}},
    "dien_thoai": {"label": "Điện thoại và phụ kiện", "kind": "retail", "orders": (6, 35), "groups": {
        "Điện thoại": [("iPhone 15 128GB", "chiếc", 19990000, []), ("Samsung Galaxy A55", "chiếc", 9490000, []),
                       ("Xiaomi Redmi Note 13", "chiếc", 4890000, []), ("OPPO A79", "chiếc", 5990000, [])],
        "Phụ kiện": [("Cáp sạc Type-C 1m", "sợi", 79000, []), ("Củ sạc nhanh 20W", "cái", 290000, []), ("Ốp lưng silicon", "cái", 59000, []),
                     ("Kính cường lực", "miếng", 69000, []), ("Tai nghe bluetooth", "cái", 450000, [])],
        "Sim thẻ": [("Sim 4G data", "cái", 120000, []), ("Thẻ cào 100k", "thẻ", 100000, [])]}},
    "sua_xe": {"label": "Sửa xe máy", "kind": "service", "orders": (10, 45), "groups": {
        "Dịch vụ": [("Thay nhớt", "lần", 30000, []), ("Bảo dưỡng tổng quát", "lần", 150000, []), ("Vá săm", "lần", 20000, []),
                    ("Rửa xe", "lần", 30000, []), ("Sửa điện", "lần", 80000, [])],
        "Phụ tùng": [("Nhớt Castrol 1L", "chai", 115000, [("thùng", 12)]), ("Lốp Michelin 90/90", "cái", 520000, []),
                     ("Má phanh", "bộ", 85000, []), ("Bugi NGK", "cái", 45000, [("hộp", 10)]), ("Săm xe", "cái", 70000, [])]}},
    "me_va_be": {"label": "Mẹ và bé", "kind": "retail", "orders": (15, 60), "groups": {
        "Sữa bột": [("Sữa Meiji 800g", "hộp", 545000, []), ("Sữa Enfamil A+ 830g", "hộp", 690000, []), ("Sữa Optimum Gold 800g", "hộp", 465000, [])],
        "Tã bỉm": [("Bỉm Bobby size M", "miếng", 5000, [("gói", 60)]), ("Bỉm Huggies size L", "miếng", 6000, [("gói", 54)])],
        "Đồ dùng": [("Bình sữa Pigeon 240ml", "cái", 215000, []), ("Khăn ướt Mamamy", "gói", 32000, [("thùng", 24)]), ("Núm ti silicon", "cái", 65000, [])],
        "Quần áo": [("Bộ body cotton", "bộ", 120000, []), ("Mũ sơ sinh", "cái", 35000, [])]}},
    "thoi_trang": {"label": "Thời trang", "kind": "retail", "orders": (10, 50), "groups": {
        "Áo": [("Áo thun basic", "cái", 159000, []), ("Áo sơ mi trắng", "cái", 299000, []), ("Áo khoác gió", "cái", 399000, [])],
        "Quần": [("Quần jeans slim", "cái", 449000, []), ("Quần short kaki", "cái", 229000, [])],
        "Váy": [("Váy hoa nhí", "cái", 359000, []), ("Chân váy xếp ly", "cái", 259000, [])],
        "Phụ kiện": [("Thắt lưng da", "cái", 199000, []), ("Tất cổ ngắn", "đôi", 25000, [("hộp", 5)])]}},
    "my_pham": {"label": "Mỹ phẩm", "kind": "retail", "orders": (10, 55), "groups": {
        "Chăm sóc da": [("Sữa rửa mặt Cetaphil 125ml", "chai", 265000, []), ("Kem chống nắng Anessa 60ml", "tuýp", 495000, []),
                        ("Nước tẩy trang Bioderma 500ml", "chai", 435000, [])],
        "Trang điểm": [("Son MAC Ruby Woo", "thỏi", 590000, []), ("Phấn phủ Innisfree", "hộp", 210000, []), ("Mascara Maybelline", "cây", 189000, [])],
        "Mặt nạ": [("Mặt nạ giấy Mediheal", "miếng", 25000, [("hộp", 10)])]}},
    "van_phong_pham": {"label": "Văn phòng phẩm", "kind": "retail", "orders": (20, 80), "groups": {
        "Bút": [("Bút bi Thiên Long TL-027", "cây", 5000, [("hộp", 20)]), ("Bút chì 2B", "cây", 4000, [("hộp", 12)]), ("Bút dạ quang", "cây", 9000, [])],
        "Giấy vở": [("Vở 96 trang", "quyển", 9000, [("lốc", 10)]), ("Giấy A4 Double A", "ram", 85000, [("thùng", 5)])],
        "Dụng cụ": [("Thước kẻ 20cm", "cái", 6000, []), ("Gôm tẩy", "cục", 4000, [("hộp", 30)]), ("Kéo văn phòng", "cái", 25000, [])]}},
    "thu_cung": {"label": "Thú cưng", "kind": "retail", "orders": (10, 45), "groups": {
        "Thức ăn": [("Hạt Royal Canin mèo 2kg", "túi", 520000, []), ("Hạt Pedigree chó 1.5kg", "túi", 165000, []), ("Pate Whiskas 80g", "gói", 15000, [("hộp", 12)])],
        "Phụ kiện": [("Cát vệ sinh mèo 5L", "bao", 95000, []), ("Vòng cổ", "cái", 60000, []), ("Đồ chơi chuột bông", "cái", 30000, [])],
        "Dịch vụ": [("Tắm cắt tỉa", "lần", 180000, []), ("Gửi thú cưng", "ngày", 120000, [])]}},
    "tiem_banh": {"label": "Tiệm bánh", "kind": "fnb", "orders": (30, 120), "groups": {
        "Bánh ngọt": [("Bánh bông lan trứng muối", "cái", 45000, []), ("Bánh su kem", "cái", 8000, [("hộp", 10)]), ("Tiramisu", "phần", 55000, [])],
        "Bánh mì": [("Bánh mì hoa cúc", "ổ", 65000, []), ("Bánh mì sandwich", "ổ", 30000, [])],
        "Bánh kem": [("Bánh kem 16cm", "cái", 250000, []), ("Bánh kem 20cm", "cái", 350000, [])],
        "Đồ uống": [("Trà sữa trân châu (M)", "ly", 35000, []), ("Nước cam ép", "ly", 30000, [])]}},
    "lau_nuong": {"label": "Quán lẩu nướng", "kind": "fnb", "orders": (25, 90), "groups": {
        "Lẩu": [("Lẩu Thái hải sản", "nồi", 299000, []), ("Lẩu bò nhúng giấm", "nồi", 259000, [])],
        "Nướng": [("Ba chỉ bò nướng", "đĩa", 119000, []), ("Sườn heo nướng", "đĩa", 99000, []), ("Mực nướng sa tế", "đĩa", 139000, [])],
        "Đồ uống": [("Bia Tiger", "lon", 22000, [("thùng", 24)]), ("Nước ngọt", "lon", 15000, []), ("Trà đá", "ly", 5000, [])],
        "Món thêm": [("Rau nhúng", "đĩa", 25000, []), ("Mì gói nhúng", "gói", 10000, [])]}},
    "hoa_tuoi": {"label": "Hoa tươi", "kind": "retail", "orders": (8, 40), "groups": {
        "Hoa bó": [("Bó hồng đỏ 20 bông", "bó", 450000, []), ("Bó hướng dương", "bó", 320000, []), ("Bó cẩm tú cầu", "bó", 380000, [])],
        "Hoa cắm": [("Giỏ hoa chúc mừng", "giỏ", 650000, []), ("Kệ hoa khai trương", "kệ", 1500000, [])],
        "Hoa cành": [("Hoa hồng Đà Lạt", "cành", 15000, [("bó", 20)]), ("Hoa cúc", "cành", 8000, [("bó", 10)])]}},
    "nong_san": {"label": "Phân bón và nông sản", "kind": "retail", "orders": (5, 30), "groups": {
        "Phân bón": [("Phân NPK 16-16-8", "kg", 18000, [("bao", 50)]), ("Phân Urê Phú Mỹ", "kg", 14000, [("bao", 50)]), ("Phân hữu cơ vi sinh", "kg", 9000, [("bao", 25)])],
        "Thuốc BVTV": [("Thuốc trừ sâu Regent", "chai", 65000, [("thùng", 20)]), ("Thuốc diệt cỏ", "chai", 85000, [])],
        "Hạt giống": [("Hạt giống rau muống", "gói", 12000, []), ("Hạt giống dưa leo", "gói", 25000, [])]}},
    "dien_may": {"label": "Điện máy gia dụng", "kind": "retail", "orders": (4, 25), "groups": {
        "Điện lạnh": [("Máy lạnh Daikin 1HP", "bộ", 9890000, []), ("Tủ lạnh Toshiba 253L", "cái", 6490000, [])],
        "Gia dụng": [("Nồi cơm điện Sharp 1.8L", "cái", 890000, []), ("Quạt đứng Senko", "cái", 420000, []), ("Ấm siêu tốc 1.7L", "cái", 290000, [])],
        "Phụ kiện": [("Ổ cắm điện Điện Quang", "cái", 85000, []), ("Bóng đèn LED 9W", "bóng", 35000, [("hộp", 10)])]}},
    "giay_dep": {"label": "Giày dép", "kind": "retail", "orders": (8, 40), "groups": {
        "Giày": [("Giày thể thao trắng", "đôi", 590000, []), ("Giày tây da", "đôi", 790000, []), ("Giày búp bê", "đôi", 320000, [])],
        "Dép": [("Dép quai hậu", "đôi", 220000, []), ("Dép tổ ong", "đôi", 45000, [])],
        "Phụ kiện": [("Xi đánh giày", "hộp", 35000, []), ("Lót giày", "đôi", 30000, [])]}},
    "tra_sua": {"label": "Trà sữa", "kind": "fnb", "orders": (50, 160), "groups": {
        "Trà sữa": [("Trà sữa truyền thống (M)", "ly", 30000, []), ("Trà sữa truyền thống (L)", "ly", 38000, []), ("Trà sữa matcha (M)", "ly", 35000, []),
                    ("Trà sữa khoai môn (L)", "ly", 42000, [])],
        "Trà trái cây": [("Trà chanh dây (M)", "ly", 30000, []), ("Trà dâu tằm (L)", "ly", 38000, [])],
        "Topping": [("Trân châu đen", "phần", 6000, []), ("Thạch phô mai", "phần", 8000, [])]}},
}
TRAIN_LIKE = {"tap_hoa", "ca_phe", "nha_thuoc", "vlxd", "lau_nuong"}   # industries the train split already covers

FIRST = ["An", "Bình", "Chi", "Dũng", "Giang", "Hà", "Hải", "Hạnh", "Hằng", "Hiếu", "Hoa", "Hoàng", "Hùng", "Hương", "Khánh", "Lan",
         "Linh", "Long", "Mai", "Minh", "My", "Nam", "Ngọc", "Nhung", "Phong", "Phương", "Quân", "Quỳnh", "Sơn", "Tâm", "Thảo",
         "Thành", "Thu", "Trang", "Trung", "Tuấn", "Uyên", "Vân", "Việt", "Vy", "Yến"]
MIDDLE = ["Văn", "Thị", "Minh", "Ngọc", "Hữu", "Thanh", "Đức", "Thu", "Quốc", "Gia", "Bảo", "Kim", "Hoài", "Anh"]
LAST = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ", "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý", "Tô", "La", "Hà"]
STAFF = ["Ngân", "Tú", "Khoa", "Trâm", "Phúc", "Duyên", "Kiệt", "Thư", "Lộc", "Nhi"]
GROUPS_CUSTOMER = ["Thân thiết", "VIP", "Khách sỉ", "Thường"]
CHANNELS = ["Tại quầy", "Facebook", "Zalo", "Shopee", "Grab"]
PAYMENTS = ["Tiền mặt", "Chuyển khoản", "Ghi nợ", "COD"]
