"""Files for the tables beyond sales (synth/world2.py): purchases, ending stock, lots, price changes, a product catalog when
the archetype has none, and the event table. One flat file per table, header names drawn from a small vocabulary, described
in the README like the train shops describe theirs. What each file is made of goes to the shop's metadata
(`extra`: file, table role, header per role, value spellings), which the question generator uses to write SQL.
"""
import pandas as pd

X = {   # role -> header spellings
    "mã phiếu nhập": ["Mã nhập hàng", "Mã đơn nhập", "Số phiếu nhập", "Mã PN"], "ngày nhập": ["Ngày nhập", "Ngày phiếu", "Thời gian nhập"],
    "nhà cung cấp": ["Nhà cung cấp", "NCC", "Tên NCC", "Đơn vị cung cấp"], "trạng thái phiếu nhập": ["Trạng thái", "Tình trạng phiếu", "TT phiếu"],
    "đơn vị nhập": ["ĐVT", "Đơn vị nhập", "ĐV nhập"], "số lượng nhập": ["SL nhập", "Số lượng nhập", "Số lượng"],
    "giá nhập": ["Giá nhập", "Đơn giá nhập", "Đơn giá"], "tiền nhập": ["Thành tiền", "Tiền nhập", "Giá trị nhập"],
    "số lô": ["Lô", "Mã lô", "Số lô"], "hạn sử dụng": ["Hạn sử dụng", "HSD", "Hạn dùng"],
    "tồn cuối kỳ": ["Tồn cuối kỳ", "Tồn cuối", "Tồn thực", "Ton kho"], "tồn lô": ["Còn tồn", "Tồn lô", "SL còn"],
    "số lượng nhập lô": ["SL nhập (ĐV cơ bản)", "Số lượng nhập lô", "SL lô"], "giá nhập cơ bản": ["Giá nhập 1 ĐV cơ bản", "Giá vốn lô", "Giá lô"],
    "ngày áp dụng": ["Ngày áp dụng", "Áp dụng từ", "Ngày đổi giá"], "giá cũ": ["Giá cũ", "Giá trước"], "giá mới": ["Giá mới", "Giá sau", "Giá bán mới"],
    "tên hàng": ["Tên hàng", "Tên hàng hóa", "Sản phẩm", "Hàng hóa", "Mặt hàng"], "mã hàng": ["Mã hàng", "Mã SKU", "Mã hàng hóa"],
    "đơn vị cơ bản": ["Đơn vị cơ bản", "ĐV cơ bản", "Đơn vị tính", "ĐVT"], "giá bán": ["Giá bán", "Giá niêm yết", "Giá hiện tại"],
    # event tables
    "mã phiếu": ["Mã phiếu", "Số phiếu", "Mã tiếp nhận", "Mã lượt"], "ngày nhận": ["Ngày nhận", "Ngày tiếp nhận", "Ngày"],
    "ngày trả": ["Ngày trả", "Ngày hoàn tất", "Ngày giao lại"], "khách": ["SĐT khách", "Điện thoại KH", "Khách"],
    "nội dung": ["Nội dung", "Việc làm", "Hạng mục", "Dịch vụ"], "loại": ["Loại phiếu", "Hình thức", "Tình trạng", "Loại"],
    "người làm": ["Kỹ thuật viên", "Thợ", "Người xử lý", "Nhân viên", "KTV"], "tiền thu": ["Phí", "Tiền thu", "Thành tiền", "Số tiền"],
    # pay-by-period books and books tied to orders
    "mã trả dần": ["Mã HĐ", "Số HĐ", "Mã hồ sơ", "Số sổ"], "ngày mua": ["Ngày mua", "Ngày lập", "Ngày"], "mã đơn bán": ["Mã đơn bán", "Số đơn", "Đơn bán", "Theo đơn"],
    "khách trả dần": ["Khách", "Tên khách", "Khách hàng"], "giá trị đơn": ["Giá trị đơn", "Tiền đơn", "Giá"], "trả trước": ["Trả trước", "Đưa trước", "Trả lúc mua"],
    "số kỳ": ["Số kỳ", "Số đợt", "Số lần trả"], "tiền mỗi kỳ": ["Mỗi kỳ", "Tiền mỗi đợt", "Trả mỗi lần"], "ngày hạn": ["Ngày hạn hằng tháng", "Hạn (ngày)", "Ngày đến hạn"],
    "ngày thu": ["Ngày thu", "Ngày khách trả", "Ngày nộp"], "tiền thu kỳ": ["Số tiền", "Tiền thu", "Đã thu"], "kỳ số": ["Kỳ", "Đợt", "Lần"],
    "hạn trả": ["Hạn trả", "Ngày đến hạn", "Hạn"], "tiền kỳ": ["Tiền phải trả", "Số tiền", "Phải thu"], "ngày trả kỳ": ["Ngày trả", "Ngày thu", "Đã trả ngày"],
    "mã dòng": ["Số phiếu", "Mã phiếu", "STT"], "ngày ghi": ["Ngày ghi", "Ngày", "Ngày lập"], "mã đơn nối": ["Mã đơn bán", "Số đơn", "Theo đơn", "Đơn bán"],
    "số tiền": ["Số tiền", "Giá trị", "Thành tiền"],
}
BOOK_FILES = {"plans": ["SoTraDan.xlsx", "HopDongTraKy.xlsx", "so_tra_dan.csv"], "plan_pay": ["SoThuTien.xlsx", "PhieuThuKy.xlsx", "thu_ky.csv"],
              "plan_sched": ["LichThu.xlsx", "LichTraKy.xlsx", "lich_thu.csv"], "deposit": ["SoDatCoc.xlsx", "PhieuCoc.xlsx", "dat_coc.csv"],
              "credit": ["SoThuDoi.xlsx", "ThuHangCu.xlsx", "thu_doi.csv"], "ship": ["SoGiaoHang.xlsx", "PhiGiaoHang.xlsx", "phi_giao.csv"],
              "gift": ["SoQuaTang.xlsx", "QuaTangKem.xlsx", "qua_tang.csv"]}
LINK_CAT = {"deposit": ["Hình thức cọc", "Hình thức", "Cọc bằng"], "credit": ["Tình trạng", "Tình trạng hàng cũ", "Chất lượng"],
            "ship": ["Bên giao", "Đơn vị giao", "Giao bởi"], "gift": ["Loại quà", "Loại", "Hình thức quà"]}
ROLE = {"mã phiếu nhập": "mã phiếu nhập", "ngày nhập": "ngày nhập", "nhà cung cấp": "nhà cung cấp", "trạng thái phiếu nhập": "trạng thái phiếu nhập",
        "đơn vị nhập": "đơn vị nhập", "số lượng nhập": "số lượng nhập", "giá nhập": "giá nhập một đơn vị, theo đơn vị nhập của dòng",
        "tiền nhập": "tiền nhập của cả dòng", "số lô": "số lô", "hạn sử dụng": "hạn sử dụng", "tồn cuối kỳ": "tồn kho",
        "tồn lô": "số còn lại của lô, theo đơn vị cơ bản", "số lượng nhập lô": "số lượng nhập của lô, theo đơn vị cơ bản",
        "giá nhập cơ bản": "giá nhập của lô cho một đơn vị cơ bản", "ngày áp dụng": "ngày bắt đầu áp dụng giá", "giá cũ": "giá trước lần đổi",
        "giá mới": "giá bán áp dụng từ ngày đó", "tên hàng": "tên hàng", "mã hàng": "mã hàng", "đơn vị cơ bản": "đơn vị cơ bản",
        "giá bán": "giá bán hiện tại", "giá vốn danh mục": "giá vốn một đơn vị cơ bản"}
FILES = {"purchases": ["DanhSachNhapHang.xlsx", "SoNhapHang.xlsx", "nhap_hang.csv", "PhieuNhap.xlsx"], "stock": ["TonKho.xlsx", "BaoCaoTonKho.xlsx", "ton_kho.csv"],
         "lots": ["TonKhoTheoLo.xlsx", "DanhSachLo.xlsx", "lo_hang.csv"], "prices": ["BangGia.xlsx", "LichSuGia.xlsx", "lich_su_gia.csv"],
         "catalog": ["DanhMuc.xlsx", "DanhMucHang.xlsx", "danh_muc.csv"],
         "ticket": ["PhieuBaoHanh.xlsx", "SoTiepNhan.xlsx", "phieu_sua_chua.csv"], "visit": ["LichHen.xlsx", "SoDichVu.xlsx", "lich_hen.csv"]}
SPELL = {"ticket": {"free": ["Bảo hành", "bảo hành", "BH"], "paid": ["Sửa tính phí", "Tính phí"]},
         "visit": {"free": ["Không đến", "Huỷ hẹn"], "paid": ["Đã đến", "Hoàn thành"]},
         "receipt": {"ok": ["Hoàn thành", "Đã nhập", "đã nhập kho"], "cancel": ["Đã huỷ", "Huỷ", "huy phieu"]}}


def extras(r) -> None:
    s, rng, f = r.s, r.rng, r.s.flags
    pick = lambda role: str(rng.choice(X[role]))                                    # noqa: E731
    with_code = r.has_catalog                                                       # exports joined by codes carry the item code

    def write(kind: str, cols: list, role_text: dict | None = None, names: list | None = None) -> None:
        fname = str(rng.choice(names or FILES[kind]))
        heads, used = {}, set()
        for role, _ in cols:                                                        # one header per role, no two columns alike
            h = r.head.get(role) if role in ("thuộc tính",) else pick(role) if role in X else r.head[role]
            while h in used:
                h = h + " " + role.split()[-1]
            used.add(h)
            heads[role] = h
        df = pd.DataFrame({heads[role]: list(v) for role, v in cols})
        df.to_csv(r.folder / fname, index=False) if fname.endswith(".csv") else df.to_excel(r.folder / fname, index=False)
        for role, h in heads.items():
            r.head.setdefault(f"{kind}:{role}", h)
        r.files.append((fname, [f"{kind}:{role}" for role, _ in cols], ""))
        r.extra_meta[kind] = {"file": fname, "cols": heads}
        for role in heads:
            from synth.render import ROLE_TEXT
            ROLE_TEXT.setdefault(f"{kind}:{role}", (role_text or {}).get(role) or ROLE.get(role, role))

    item = lambda df: ([("mã hàng", df["code"])] if with_code else []) + [("tên hàng", df["name"])]   # noqa: E731

    if f.get("purchases"):
        p = s.extra["purchases"]
        st = {"hoàn thành": list(rng.choice(SPELL["receipt"]["ok"], 2, replace=False)), "huỷ": list(rng.choice(SPELL["receipt"]["cancel"], 2, replace=False))}
        cols = [("mã phiếu nhập", p["receipt_id"]), ("ngày nhập", p["date"]), ("nhà cung cấp", p["supplier"]),
                ("trạng thái phiếu nhập", [str(rng.choice(st[x])) for x in p["status"]])] + item(p) + \
               [("đơn vị nhập", p["unit"]), ("số lượng nhập", p["qty"]), ("giá nhập", p["unit_cost"]), ("tiền nhập", p["amount"])]
        if f.get("lots") and rng.random() < 0.5:
            cols += [("số lô", p["lot"]), ("hạn sử dụng", p["expiry"])]
        write("purchases", cols)
        r.extra_meta["purchases"]["status"] = st
        h = r.extra_meta["purchases"]["cols"]
        r.notes.append(f"**{h['giá nhập']}** là giá nhập một đơn vị, theo đơn vị nhập của dòng; **{h['tiền nhập']}** là tiền nhập của cả dòng. "
                       f"Phiếu nhập ghi {' hoặc '.join('`' + v + '`' for v in st['huỷ'])} là phiếu huỷ.")
    if f.get("stock"):
        t = s.extra["stock"]
        write("stock", item(t) + [("đơn vị cơ bản", t["base_unit"]), ("tồn cuối kỳ", t["ending"])])
        r.notes.append(f"**{r.extra_meta['stock']['cols']['tồn cuối kỳ']}** là tồn cuối ngày cuối có dữ liệu, theo đơn vị cơ bản.")
    if f.get("lots"):
        t = s.extra["lots"]
        write("lots", [("số lô", t["lot"])] + item(t) + [("hạn sử dụng", t["expiry"]), ("số lượng nhập lô", t["qty_base"]), ("tồn lô", t["remaining"]),
                                                           ("giá nhập cơ bản", [round(v, 2) for v in t["unit_cost_base"]])])
        h = r.extra_meta["lots"]["cols"]
        r.notes.append(f"Mỗi dòng nhập là một lô, số lô không trùng. **{h['tồn lô']}** là số còn lại của lô cuối ngày cuối có dữ liệu, theo đơn vị cơ bản.")
    if f.get("price_history"):
        t = s.extra["prices"]
        write("prices", item(t) + [("ngày áp dụng", t["date"]), ("giá cũ", t["old"]), ("giá mới", t["new"])])
        r.notes.append("Bảng giá ghi mỗi lần đổi giá bán một dòng; giá là giá một đơn vị cơ bản.")
    if not r.has_catalog and (f.get("attr") or f.get("stock")):
        p = s.products
        cols = [("tên hàng", p["name"]), ("đơn vị cơ bản", p["base_unit"]), ("giá bán", p["price"])]
        if f.get("stock"):
            cols.append(("giá vốn danh mục", p["cost"]))
        if f.get("attr"):
            r.head["thuộc tính"] = s.extra["attr_name"]
            cols.append(("thuộc tính", p["attr"]))
        write("catalog", cols, {"thuộc tính": s.extra["attr_name"].lower() + " của mặt hàng (để trống nếu không ghi)"})
    elif r.has_catalog:
        heads = {role: r.head[role] for role in ("mã hàng", "tên hàng", "đơn vị cơ bản", "giá bán")}
        if f.get("stock"):
            heads["giá vốn danh mục"] = r.head["giá vốn danh mục"]
        if f.get("attr"):
            heads["thuộc tính"] = r.head["thuộc tính"]
        r.extra_meta["catalog"] = {"file": "DanhMucHangHoa.xlsx", "cols": heads}
    if f.get("attr"):
        from synth.render import ROLE_TEXT
        ROLE_TEXT["thuộc tính"] = s.extra["attr_name"].lower() + " của mặt hàng (để trống nếu không ghi)"
    if f.get("events"):
        ev = s.extra["events"]
        flavor, e = ev["flavor"], ev["df"]
        sp = {"free": list(rng.choice(SPELL[flavor]["free"], 2, replace=False)), "paid": list(rng.choice(SPELL[flavor]["paid"], 2, replace=False))}
        cols = [("mã phiếu", e["eid"]), ("ngày nhận", e["date"])]
        if flavor == "ticket":
            cols.append(("ngày trả", e["date2"]))
        cols += [("khách", e["entity"]), ("nội dung", e["what"]), ("loại", [str(rng.choice(sp["free" if x else "paid"])) for x in e["free"]]),
                 ("người làm", e["staff"]), ("tiền thu", e["amount"])]
        grain = "một phiếu tiếp nhận" if flavor == "ticket" else "một lượt hẹn"
        write(flavor, cols, {"mã phiếu": "mã " + ("phiếu tiếp nhận" if flavor == "ticket" else "lượt hẹn"),
                             "ngày nhận": "ngày nhận máy, đồ" if flavor == "ticket" else "ngày hẹn", "ngày trả": "ngày trả lại cho khách (trống nếu chưa trả)",
                             "khách": "số điện thoại của khách", "nội dung": "việc đã làm", "loại": "loại phiếu" if flavor == "ticket" else "tình trạng lượt hẹn",
                             "người làm": "người thực hiện", "tiền thu": "tiền thu của khách (0 nếu không thu)"})
        r.extra_meta[flavor]["spell"] = sp
        r.extra_meta["events"] = flavor
        free = " hoặc ".join("`" + v + "`" for v in sp["free"])
        r.notes.append(f"Mỗi dòng `{r.extra_meta[flavor]['file']}` là {grain}. Dòng ghi {free} không thu tiền.")
    if f.get("plans"):
        pl = s.extra["plans"]
        P, S, plan, pw = pl["plans"], pl["sched"], pl["plan"], pl["p"]
        cols = [("mã trả dần", P["pid"]), ("ngày mua", P["date"]), ("mã đơn bán", P["oid"]), ("khách trả dần", P["cname"]), ("giá trị đơn", P["total"]),
                ("trả trước", P["down"]), ("số kỳ", P["n"])]
        text = {"mã trả dần": f"mã của đơn {plan}", "ngày mua": "ngày mua", "mã đơn bán": "mã đơn ở sổ bán", "khách trả dần": "tên khách",
                "giá trị đơn": "tiền cả đơn", "trả trước": "tiền trả lúc mua", "số kỳ": f"số {pw} phải trả", "tiền mỗi kỳ": f"tiền mỗi {pw}",
                "ngày hạn": f"ngày trong tháng là hạn của mỗi {pw}"}
        if pl["layout"] == "two":
            cols += [("tiền mỗi kỳ", P["per"]), ("ngày hạn", P["day"])]
        write("plans", cols, text, BOOK_FILES["plans"])
        h = r.extra_meta["plans"]["cols"]
        r.head["mã trả dần (sổ thu)"] = h["mã trả dần"]
        ROLE["mã trả dần (sổ thu)"] = f"mã của đơn {plan}"
        note = (f"`{r.extra_meta['plans']['file']}`: mỗi dòng là một đơn {plan}, nối với sổ bán qua **{h['mã đơn bán']}**; đơn {plan} vẫn ghi đủ tiền ở sổ bán. "
                f"**{h['giá trị đơn']}** là tiền cả đơn, **{h['trả trước']}** là tiền trả lúc mua, phần còn lại chia đều cho **{h['số kỳ']}** {pw}. ")
        if pl["layout"] == "two":
            paid = S.dropna(subset=["paid"])
            write("plan_pay", [("mã trả dần (sổ thu)", paid["pid"]), ("ngày thu", paid["paid"]), ("tiền thu kỳ", paid["amount"])],
                  {"ngày thu": f"ngày khách trả một {pw}", "tiền thu kỳ": "số tiền thu lần đó"}, BOOK_FILES["plan_pay"])
            note += (f"Mỗi {pw} phải trả **{h['tiền mỗi kỳ']}**; {pw} thứ nhất có hạn vào ngày **{h['ngày hạn']}** của tháng ngay sau tháng mua, mỗi {pw} sau có hạn "
                     f"cùng ngày đó của tháng kế tiếp. `{r.extra_meta['plan_pay']['file']}`: mỗi lần khách trả một {pw} là một dòng.")
        else:
            write("plan_sched", [("mã trả dần (sổ thu)", S["pid"]), ("kỳ số", S["k"]), ("hạn trả", S["due"]), ("tiền kỳ", S["amount"]), ("ngày trả kỳ", S["paid"])],
                  {"kỳ số": f"số thứ tự của {pw}", "hạn trả": f"ngày hạn của {pw}", "tiền kỳ": f"tiền phải trả của {pw}",
                   "ngày trả kỳ": "ngày khách trả (trống nếu chưa trả)"}, BOOK_FILES["plan_sched"])
            hs = r.extra_meta["plan_sched"]["cols"]
            note += (f"`{r.extra_meta['plan_sched']['file']}`: mỗi dòng là một {pw} của một đơn {plan}, có hạn trả và số tiền; **{hs['ngày trả kỳ']}** để trống là chưa trả.")
        r.extra_meta["plans"]["layout"] = pl["layout"]
        r.notes.append(note)
    if f.get("links"):
        lk = s.extra["links"]
        e, flavor = lk["df"], lk["flavor"]
        r.head["phân loại sổ phụ"] = str(rng.choice(LINK_CAT[flavor]))
        write("links", [("mã dòng", e["lid"]), ("ngày ghi", e["date"]), ("mã đơn nối", e["oid"]), ("phân loại sổ phụ", e["cat"]), ("số tiền", e["amount"])],
              {"mã dòng": "mã của dòng", "ngày ghi": "ngày ghi khoản này", "mã đơn nối": "mã đơn ở sổ bán mà khoản này thuộc về",
               "phân loại sổ phụ": lk["cat_name"], "số tiền": f"tiền {lk['w']}"}, BOOK_FILES[flavor])
        h = r.extra_meta["links"]["cols"]
        r.extra_meta["links"]["flavor"] = flavor
        r.notes.append(f"`{r.extra_meta['links']['file']}` ({lk['book']}): mỗi dòng là một khoản {lk['w']}, nối với sổ bán qua **{h['mã đơn nối']}**; một đơn có thể có "
                       f"nhiều dòng. Tiền ở sổ bán là tiền đủ của đơn, chưa trừ khoản này.")
