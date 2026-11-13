"""Render a simulated shop into the kinds of files the competition ships, plus its README.

Three archetypes, all seen in the train split: a flat file (one row per order line), a set of POS exports joined by codes,
and a hand-kept ledger (one sheet per month, two header rows, money written as '55.000đ' or '55k', dates as text or Excel
serials, order-level cells filled only on an order's first line, duplicated lines inside an order). Header names come from
the train READMEs. The messiness never changes the answers: the rulebook says how to read each of these (section 8).
"""
from datetime import date, datetime
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from synth.world import Shop

HEADERS = {   # role -> header spellings (train READMEs; staff, channel and payment are not in train and get plain names)
    "mã đơn": ["Mã đơn hàng", "Mã hóa đơn", "Mã chứng từ", "Số CT", "Số phiếu", "Mã bill", "Mã đơn"],
    "ngày bán": ["Ngày", "Ngày đặt hàng", "Thời gian", "Ngày giao dịch", "Ngày tháng", "Ngày bán"],
    "giờ đặt đơn": ["Giờ", "Giờ đặt", "Thời điểm"],
    "tên hàng": ["Tên sản phẩm", "Tên hàng hóa", "Tên hàng", "Hàng", "Product", "Nội dung"],
    "mã hàng": ["Mã hàng", "Mã SKU", "Mã hàng hóa", "ID SP", "SKU", "Mã mặt hàng"],
    "nhóm hàng": ["Nhóm hàng hóa", "Loại sản phẩm", "Nhom hang", "Danh mục", "Nhóm hàng"],
    "đơn vị tính": ["Đơn vị tính", "ĐVT", "Đơn vị (tính)", "Unit", "DonVi", "DVT"],
    "đơn vị cơ bản": ["Đơn vị cơ bản", "Đơn vị gốc", "ĐV nhỏ", "Đơn vị"],
    "đơn vị lớn": ["Đơn vị quy đổi", "ĐV nhập", "ĐV lớn"],
    "hệ số": ["Tỉ lệ quy đổi", "Hệ số", "Hệ số QĐ"],
    "số lượng": ["Số lượng", "Số lượng sản phẩm", "so_luong", "SL bán", "Số lượng bán", "SL"],
    "số lượng quy đổi": ["Số lượng quy đổi", "Số lượng lẻ", "Quy ra ĐV lẻ", "SL lẻ", "SL quy đổi"],
    "đơn giá": ["Đơn giá", "Giá sản phẩm", "Giá lẻ", "Giá 1 đv", "Price", "Giá đơn vị"],
    "giảm giá": ["Giảm giá", "Chiết khấu", "CK"],
    "thành tiền": ["Thành tiền", "Tổng tiền", "Trị giá", "Amount", "Thành tiền (đ)", "T.Tiền"],
    "trạng thái đơn": ["Trạng thái", "Trạng thái đơn hàng", "Status", "Trạng thái xử lý", "Trạng thái đơn"],
    "tên khách": ["Khách hàng", "Tên khách hàng", "Khách", "Tên khách", "Bên mua"],
    "mã khách": ["Mã KH", "Mã khách hàng", "Số KH", "ma_kh", "Customer ID"],
    "nhóm khách": ["Nhóm khách", "Loại khách", "Hạng thành viên"],
    "số điện thoại": ["Điện thoại", "Phone", "Di động", "Số điện thoại"],
    "nhân viên": ["Nhân viên", "NV bán", "Thu ngân", "Người bán"],
    "kênh bán": ["Kênh bán", "Kênh", "Nguồn đơn"],
    "thanh toán": ["Thanh toán", "Hình thức TT", "PTTT"],
    "giá bán": ["Giá bán", "Giá", "Giá niêm yết", "Giá hiện tại"],
    "giá vốn": ["Giá vốn", "Giá vốn lúc bán", "GiaVon"],
    "giá vốn danh mục": ["Giá vốn (ĐV cơ bản)", "Giá vốn hàng", "Giá vốn 1 ĐV"],
}
ROLE_TEXT = {"mã đơn": "mã đơn", "ngày bán": "ngày bán", "giờ đặt đơn": "giờ đặt đơn", "tên hàng": "tên hàng", "mã hàng": "mã hàng",
             "nhóm hàng": "nhóm hàng", "đơn vị tính": "đơn vị tính", "đơn vị cơ bản": "đơn vị cơ bản", "đơn vị lớn": "đơn vị lớn",
             "hệ số": "hệ số: số đơn vị cơ bản trong 1 đơn vị lớn", "số lượng": "số lượng",
             "số lượng quy đổi": "số lượng đã đổi ra đơn vị cơ bản", "đơn giá": "đơn giá", "giảm giá": "giảm giá trên dòng",
             "thành tiền": "thành tiền sau giảm", "trạng thái đơn": "trạng thái đơn",
             "tên khách": "tên khách (Khách lẻ nếu không lưu)", "mã khách": "mã khách", "nhóm khách": "nhóm khách",
             "số điện thoại": "số điện thoại", "nhân viên": "nhân viên bán", "kênh bán": "kênh bán", "thanh toán": "hình thức thanh toán",
             "giá bán": "giá bán hiện tại", "giá vốn": "giá vốn một đơn vị bán (cùng đơn vị với đơn giá)",
             "giá vốn danh mục": "giá vốn một đơn vị cơ bản"}
STATUS = {"hoàn thành": ["Hoàn thành", "Đã hoàn thành", "đã xong", "completed", "Đã giao", "đã lấy", "ok"],
          "huỷ": ["Huỷ", "Đã huỷ", "huy", "cancelled", "Khách huỷ"],
          "trả hàng": ["Trả hàng", "Hoàn trả", "returned", "hang tra ve", "Khách trả"],
          "đang giao": ["Đang giao", "shipping", "đang ship", "Đang vận chuyển"]}
WALK_IN = ["Khách lẻ", "Khách lẻ", "lẻ", "kl"]


class Renderer:
    def __init__(self, shop: Shop, rng: np.random.Generator, folder: Path):
        self.s, self.rng, self.folder = shop, rng, folder
        self.head = {role: str(rng.choice(v)) for role, v in HEADERS.items()}
        self.status = {k: list(rng.choice(v, size=min(len(v), int(rng.integers(1, 4))), replace=False)) for k, v in STATUS.items()}
        self.walk_in = list(rng.choice(WALK_IN, size=int(rng.integers(1, 3)), replace=False))
        self.date_fmt = str(rng.choice(["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d %H:%M:%S"]))
        self.files: list[tuple[str, list[str], str]] = []      # (file name, roles, note)
        self.has_catalog, self.notes, self.extra_meta = False, [], {}

    def status_text(self, st: str) -> str:
        return str(self.rng.choice(self.status[st]))

    def cust_text(self, name) -> str:
        if name is None or (isinstance(name, float) and np.isnan(name)):
            return str(self.rng.choice(self.walk_in))
        if not self.s.flags["customer_code"] and self.rng.random() < 0.08:   # same customer, another spelling (rulebook 3)
            return str(name).lower() if self.rng.random() < 0.5 else f" {name}"
        return str(name)

    def joined(self) -> pd.DataFrame:
        return self.s.lines.merge(self.s.orders, on="order_id").sort_values(["date", "time", "order_id", "line"])

    # ---------- archetypes ----------

    def flat(self) -> str:
        f, d = self.s.flags, self.joined()
        cols = [("mã đơn", d["order_id"]), ("ngày bán", [t.strftime(self.date_fmt) if "%H" in self.date_fmt else x.strftime(self.date_fmt)
                                                          for x, t in zip(d["date"], d["time"])])]
        if f["time"] and "%H" not in self.date_fmt:
            cols.append(("giờ đặt đơn", [t.strftime("%H:%M") for t in d["time"]]))
        cols += [("tên hàng", d["name"])]
        if f["group"]:
            cols.append(("nhóm hàng", d["group"]))
        cols += [("đơn vị tính", d["unit"]), ("số lượng", d["qty"])]
        if self.rng.random() < 0.5:
            cols.append(("số lượng quy đổi", d["qty_base"]))
        cols += [("đơn giá", d["unit_price"]), ("thành tiền", d["amount"])]
        if f.get("cost"):
            cols.append(("giá vốn", d["unit_cost"]))
        if f["status"]:
            cols.append(("trạng thái đơn", [self.status_text(x) for x in d["status"]]))
        cols.append(("tên khách", [self.cust_text(x) for x in d["cname"]]))
        if f["customer_code"]:
            cols.append(("mã khách", d["ckey"].fillna("")))
        for flag, role, col in (("staff", "nhân viên", "staff"), ("channel", "kênh bán", "channel"), ("payment", "thanh toán", "payment")):
            if f[flag]:
                cols.append((role, d[col]))
        order = list(self.rng.permutation(len(cols)))
        cols = [cols[i] for i in order]
        df = pd.DataFrame({self.head[r]: list(v) for r, v in cols})
        name = str(self.rng.choice(["ban_hang", "don_hang", "lich_su_ban", "BanHang", "du_lieu_ban"]))
        if self.rng.random() < 0.6:
            df.to_csv(self.folder / f"{name}.csv", index=False)
            fname = f"{name}.csv"
        else:
            df.to_excel(self.folder / f"{name}.xlsx", index=False)
            fname = f"{name}.xlsx"
        self.files.append((fname, [r for r, _ in cols], ""))
        self.conversions_in_readme = True
        return "Một file phẳng, mỗi dòng một bản ghi (một dòng hàng trong đơn)."

    def pos(self) -> str:
        f, s = self.s.flags, self.s
        o = s.orders.sort_values(["date", "time"])
        ocols = [("mã đơn", o["order_id"]), ("ngày bán", [t.strftime("%Y-%m-%d %H:%M:%S") if f["time"] else x.strftime(self.date_fmt)
                                                          for x, t in zip(o["date"], o["time"])])]
        if f["customer_code"]:
            ocols.append(("mã khách", o["ckey"].fillna("")))
        else:
            ocols.append(("tên khách", [self.cust_text(x) for x in o["cname"]]))
        if f["status"]:
            ocols.append(("trạng thái đơn", [self.status_text(x) for x in o["status"]]))
        for flag, role, col in (("staff", "nhân viên", "staff"), ("channel", "kênh bán", "channel"), ("payment", "thanh toán", "payment")):
            if f[flag]:
                ocols.append((role, o[col]))
        self._xlsx("DanhSachDonHang.xlsx", ocols)
        lines = s.lines.copy()
        lcols = [("mã đơn", lines["order_id"]), ("mã hàng", lines["code"]), ("đơn vị tính", lines["unit"]), ("số lượng", lines["qty"]),
                 ("đơn giá", lines["unit_price"])]
        if lines["discount"].any() and self.rng.random() < 0.5:
            lcols.append(("giảm giá", lines["discount"]))
        lcols.append(("thành tiền", lines["amount"]))
        if f.get("cost"):
            lcols.append(("giá vốn", lines["unit_cost"]))
        self._xlsx("ChiTietDonHang.xlsx", lcols)
        p = s.products
        pcols = [("mã hàng", p["code"]), ("tên hàng", p["name"])] + ([("nhóm hàng", p["group"])] if f["group"] else []) + \
                [("đơn vị cơ bản", p["base_unit"]), ("giá bán", p["price"])]
        self.catalog_cols(pcols)
        self._xlsx("DanhMucHangHoa.xlsx", pcols)
        self.has_catalog = True
        conv = [(r["name"], u, k, r["base_unit"]) for _, r in p.iterrows() for u, k in r["units"][1:]]
        if conv:
            self._xlsx("QuyDoiDonVi.xlsx", [("tên hàng", [c[0] for c in conv]), ("đơn vị lớn", [c[1] for c in conv]),
                                            ("hệ số", [c[2] for c in conv]), ("đơn vị cơ bản", [c[3] for c in conv])])
        if f["customer_code"]:
            c = s.customers
            ccols = [("mã khách", c["code"]), ("tên khách", c["name"]), ("số điện thoại", c["phone"])]
            if f["customer_group"]:
                ccols.append(("nhóm khách", c["group"]))
            self._xlsx("DanhSachKhachHang.xlsx", ccols)
        self.conversions_in_readme = False
        return "File Excel xuất từ phần mềm bán hàng, mỗi danh sách một file (danh sách đơn, chi tiết đơn, hàng hóa...) nối với nhau qua mã."

    def ledger(self) -> str:
        # A hand-kept book writes no customer codes, product groups, staff, channel or payment: the answers must follow
        # (customers are told apart by name, rulebook 3; questions on the rest become unanswerable, rulebook 7).
        s = self.s
        for k in ("customer_code", "group", "staff", "channel", "payment", "customer_group", "cost"):
            s.flags[k] = False
        s.orders["ckey"] = s.orders["cname"]
        f, d = s.flags, self.joined()
        roles = ["tên hàng", "số lượng", "đơn vị tính", "thành tiền"] + (["trạng thái đơn"] if f["status"] else []) + \
                ["đơn giá"] + (["giờ đặt đơn"] if f["time"] else []) + ["mã đơn", "ngày bán", "tên khách"]
        group_row = ["Hàng", None, None, "Tiền", None, "Đơn"] + [None] * (len(roles) - 6)
        sheet_fmt = str(self.rng.choice(["Thang {m:02d} {y}", "T{m:02d}-{y}"]))
        books = {}
        for (y, m), chunk in d.groupby([d["date"].map(lambda x: x.year), d["date"].map(lambda x: x.month)], sort=True):
            wb = books.setdefault(y, openpyxl.Workbook())
            ws = wb.create_sheet(sheet_fmt.format(m=m, y=y))
            ws.append(group_row)
            ws.append([self.head[r] for r in roles])
            prev = None
            for r in chunk.itertuples():
                first = r.order_id != prev
                prev = r.order_id
                row = []
                for role in roles:
                    if role == "tên hàng":
                        row.append(r.name)
                    elif role == "số lượng":
                        row.append(int(r.qty))
                    elif role == "đơn vị tính":
                        row.append(r.unit)
                    elif role in ("thành tiền", "đơn giá"):
                        v = int(r.amount if role == "thành tiền" else r.unit_price)
                        row.append(self._money(v))
                    elif role == "trạng thái đơn":
                        row.append(self.status_text(r.status) if first else None)
                    elif role == "giờ đặt đơn":
                        row.append(r.time if first else None)
                    elif role == "mã đơn":
                        row.append(r.order_id)
                    elif role == "ngày bán":
                        row.append((r.date.strftime("%d/%m/%Y") if self.rng.random() < 0.6 else (r.date - date(1899, 12, 30)).days) if first else None)
                    elif role == "tên khách":
                        row.append(self.cust_text(r.cname) if first else None)
                ws.append(row)
                if not first and self.rng.random() < 0.02:     # export error: the same line twice in one order (counts once)
                    ws.append(row)
        names = []
        for y, wb in books.items():
            del wb[wb.sheetnames[0]]
            fname = f"SoBanHang_{y}.xlsx"
            wb.save(self.folder / fname)
            names.append(fname)
        for fname in names:
            self.files.append((fname, roles, "two header rows"))
        self.conversions_in_readme = True
        return "Sổ bán hàng ghi tay chuyển sang Excel, mỗi tháng một sheet."

    def catalog_cols(self, pcols: list) -> None:
        """Catalog columns beyond the basics: cost per base unit (stock value) and the shop's attribute column."""
        p, f = self.s.products, self.s.flags
        if f.get("stock") and "cost" in p:
            pcols.append(("giá vốn danh mục", p["cost"]))
        if f.get("attr"):
            self.head["thuộc tính"] = self.s.extra["attr_name"]
            pcols.append(("thuộc tính", p["attr"]))

    # ---------- helpers ----------

    def _money(self, v: int):
        u = self.rng.random()
        if u < 0.45:
            return f"{v:,}".replace(",", ".") + "đ"
        if u < 0.7 and v % 1000 == 0:
            return f"{v // 1000}k"
        return v

    def _xlsx(self, fname: str, cols: list):
        df = pd.DataFrame({self.head[r]: list(v) for r, v in cols})
        df.to_excel(self.folder / fname, index=False)
        self.files.append((fname, [r for r, _ in cols], ""))

    def readme(self, description: str, detail: str) -> str:
        s = self.s
        out = [f"# Dữ liệu {s.sid}", "", description, "", "## File"] + [f"- `{n}` ({n.rsplit('.', 1)[1]})" for n, _, _ in self.files] + [""]
        if detail != "none":
            for name, roles, note in self.files:
                keep = roles if detail == "full" else [r for r in roles if self.rng.random() < 0.6]
                if not keep:
                    continue
                out.append(f"## Cột trong `{name}`" + (" (hai dòng tiêu đề)" if note else ""))
                out += [f"- **{self.head[r]}**: {ROLE_TEXT.get(r, r)}" for r in keep] + [""]
        out += ["## Quy ước", f"- Dữ liệu có đến hết {s.end:%d/%m/%Y}."]
        conv = [(r["name"], u, k, r["base_unit"]) for _, r in s.products.iterrows() for u, k in r["units"][1:]]
        if conv and self.conversions_in_readme:
            out.append("- **Quy đổi đơn vị**: số lượng ghi theo đơn vị của dòng:")
            out += [f"  - {n}: 1 {u} = {k} {b}" for n, u, k, b in conv]
        if s.flags["status"] and self.rng.random() < 0.5:
            out.append("- Trạng thái đơn: " + "; ".join(f"{'/'.join(v)} = {k}" for k, v in self.status.items()) + ".")
        out.append("- Khách không lưu tên ghi là " + " hoặc ".join(f"`{w}`" for w in dict.fromkeys(self.walk_in)) + ".")
        out += [f"- {n}" for n in self.notes]
        return "\n".join(out) + "\n"


def render(shop: Shop, rng: np.random.Generator, folder: Path) -> tuple[str, dict]:
    """Write the shop's files and README. Returns the archetype and what the files are made of (headers per role, status
    spellings, walk-in labels, files and their roles), which synth/goldviews.py needs and the model never sees."""
    folder.mkdir(parents=True, exist_ok=True)
    r = Renderer(shop, rng, folder)
    kind = str(rng.choice(["flat", "pos", "ledger"], p=[0.35, 0.4, 0.25]))
    description = getattr(r, kind)()
    if shop.extra.get("attr_name"):                         # world v2: the tables beyond sales
        from synth.render2 import extras
        extras(r)
    detail = str(rng.choice(["full", "partial", "none"], p=[0.6, 0.25, 0.15]))
    (folder / "README.md").write_text(r.readme(description, detail), encoding="utf-8")
    meta = {"archetype": kind, "readme": detail, "head": r.head, "status": r.status, "walk_in": r.walk_in,
            "files": [{"name": n, "roles": roles} for n, roles, _ in r.files], "flags": shop.flags, "extra": r.extra_meta,
            "products": [{"code": p["code"], "name": p["name"], "group": p["group"], "base_unit": p["base_unit"],
                          "units": [[u, int(k)] for u, k in p["units"]]} for _, p in shop.products.iterrows()]}
    meta["flags"] = {k: bool(v) for k, v in shop.flags.items()}
    return kind, meta
