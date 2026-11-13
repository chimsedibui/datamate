"""Randomised rule definitions for synthetic splits.

The public rulebook showed that a new split brings new definitions: named rankings, thresholds, rates, attribute rules and
event tables with their own counting rule. A model that memorised the train rulebook answers those from the meaning of the
words. Here every synthetic split draws its own rule set: the same kind of term gets a different name and a different
definition from split to split, written into that split's rulebook, and the answers follow the definition. Validation splits
use term names and some definitions that no training split has.

    rules = choose(rng, val=False)
    text = rulebook(train_rulebook_text, rules)
"""
import re

import numpy as np

LIB = {
    "cust_rank": {
        "terms": {"train": ["Khách ruột nhất", "Khách chịu chi nhất", "Khách bền bỉ nhất", "Khách số một", "Khách đáng giá nhất", "Khách cứng nhất"],
                  "val": ["Khách tủ nhất", "Khách xịn nhất", "Khách gắn bó nhất"]},
        "variants": {"days": "khách (xác định như dòng Xác định khách) có **nhiều ngày có mua nhất** trong kỳ",
                     "avg": "khách có **giá trị trung bình một đơn** cao nhất trong kỳ, chỉ xét khách có **từ {n} đơn trở lên** trong kỳ",
                     "orders": "khách có **nhiều đơn nhất** trong kỳ",
                     "max_order": "khách có **một đơn giá trị cao nhất** trong kỳ",
                     "qty": "khách mua **nhiều số lượng nhất** trong kỳ, quy về đơn vị cơ bản",
                     "items": "khách mua **nhiều mặt hàng khác nhau nhất** trong kỳ"},
        "val_only": ["items"], "section": "3"},
    "item_rank": {
        "terms": {"train": ["Mặt hàng quốc dân", "Mặt hàng phủ sóng nhất", "Mặt hàng tủ", "Mặt hàng đinh", "Mặt hàng ruột của quán"],
                  "val": ["Mặt hàng át chủ bài", "Mặt hàng được chuộng nhất"]},
        "variants": {"orders": "mặt hàng có mặt trong **nhiều đơn nhất** trong kỳ, không theo số lượng",
                     "customers": "mặt hàng được **nhiều khách khác nhau mua nhất** trong kỳ, không tính khách lẻ",
                     "days": "mặt hàng có **nhiều ngày bán được nhất** trong kỳ",
                     "amount": "mặt hàng có **doanh thu cao nhất** trong kỳ, không theo số lượng"},
        "val_only": ["days"], "section": "3"},
    "day_rank": {
        "terms": {"train": ["Ngày chạy đơn nhất", "Ngày nhộn nhịp nhất", "Ngày vàng", "Ngày bận nhất"], "val": ["Ngày đỉnh nhất", "Ngày tấp nập nhất"]},
        "variants": {"orders": "ngày có **nhiều đơn nhất** trong kỳ", "customers": "ngày có **nhiều khách khác nhau mua nhất** trong kỳ, không tính khách lẻ",
                     "qty": "ngày bán được **nhiều số lượng nhất** trong kỳ, quy về đơn vị cơ bản", "amount": "ngày có **doanh thu cao nhất** trong kỳ"},
        "val_only": ["customers"], "section": "3"},
    "hour_rank": {
        "terms": {"train": ["Giờ vàng", "Khung giờ đông nhất", "Giờ chạy đơn"], "val": ["Giờ nóng", "Khung giờ đỉnh"]},
        "variants": {"orders": "khung **một giờ tròn** theo giờ tạo đơn (từ h giờ đến trước h + 1 giờ) có **nhiều đơn nhất** trong kỳ. Đáp án là số h",
                     "amount": "khung **một giờ tròn** theo giờ tạo đơn (từ h giờ đến trước h + 1 giờ) có **doanh thu cao nhất** trong kỳ. Đáp án là số h"},
        "val_only": [], "section": "3"},
    "staff_rank": {
        "terms": {"train": ["Nhân viên chuyên cần nhất", "Nhân viên mát tay nhất", "Nhân viên của kỳ", "Nhân viên cày nhất"],
                  "val": ["Nhân viên xông xáo nhất", "Nhân viên chắc tay nhất"]},
        "variants": {"days": "nhân viên có **nhiều ngày có đơn nhất** trong kỳ", "orders": "nhân viên có **nhiều đơn nhất** trong kỳ, không theo doanh thu",
                     "avg": "nhân viên có **giá trị trung bình một đơn** cao nhất trong kỳ",
                     "customers": "nhân viên bán cho **nhiều khách khác nhau nhất** trong kỳ, không tính khách lẻ"},
        "val_only": ["customers"], "section": "3"},
    "big_order": {
        "terms": {"train": ["Đơn lớn", "Đơn sỉ", "Đơn khủng", "Đơn to"], "val": ["Đơn bự", "Đơn đáng kể"]},
        "variants": {"amount": "đơn (tính như §1) có doanh thu **từ {x} đồng trở lên**", "lines": "đơn (tính như §1) có **từ {k} dòng hàng trở lên**",
                     "above": "đơn (tính như §1) có doanh thu **trên {x} đồng**, không gồm đúng {x} đồng"},
        "val_only": ["above"], "section": "1"},
    "loyal": {
        "terms": {"train": ["Khách thân", "Khách mối", "Khách ruột", "Khách hay ghé"], "val": ["Khách quen mặt", "Khách thường xuyên"]},
        "variants": {"orders": "khách có **từ {n} đơn trở lên** trong kỳ", "days": "khách có mua vào **từ {n} ngày khác nhau trở lên** trong kỳ",
                     "amount": "khách có doanh thu trong kỳ **từ {x} đồng trở lên**"},
        "val_only": ["amount"], "section": "3"},
    "weekday_avg": {
        "terms": {"train": ["Trung bình mỗi ngày thứ X của kỳ"], "val": ["Trung bình mỗi ngày thứ X của kỳ"]},
        "variants": {"all_days": "chia cho **số ngày thứ X** trong kỳ, kể cả ngày không có đơn, phiếu",
                     "active_days": "chia cho **số ngày thứ X có ít nhất một đơn, phiếu** trong kỳ; ngày thứ X không có đơn, phiếu thì không tính"},
        "val_only": [], "section": "2"},
    "bad_rate": {
        "terms": {"train": ["Tỉ lệ đơn hỏng", "Tỉ lệ đơn rớt", "Tỉ lệ đơn không thành"], "val": ["Tỉ lệ đơn lỗi", "Tỉ lệ đơn bể"]},
        "variants": {"both": "phần trăm đơn **huỷ hoặc trả hàng** trong mọi đơn đặt trong kỳ (mọi trạng thái)",
                     "returned": "phần trăm đơn **trả hàng** (không tính đơn huỷ) trong mọi đơn đặt trong kỳ (mọi trạng thái)",
                     "of_done": "số đơn **huỷ hoặc trả hàng** chia cho số đơn **không huỷ, không trả** đặt trong kỳ, tính ra phần trăm"},
        "val_only": ["of_done"], "section": "1"},
    "new_cust": {
        "terms": {"train": ["Khách mới", "Khách lần đầu", "Khách mới toanh"], "val": ["Khách vừa có", "Khách mới phát sinh"]},
        "variants": {"first": "khách có **đơn đầu tiên** (tính như §1, trong toàn bộ dữ liệu) nằm trong kỳ"},
        "val_only": [], "section": "3"},
    "margin": {
        "terms": {"train": ["Biên lãi gộp", "Tỉ suất lãi gộp", "Tỉ lệ lãi"], "val": ["Biên lợi nhuận gộp", "Mức lời"]},
        "variants": {"revenue": "lãi gộp (§5) chia cho **doanh thu** cùng kỳ, tính ra phần trăm", "cost": "lãi gộp (§5) chia cho **giá vốn hàng đã bán** cùng kỳ, tính ra phần trăm"},
        "val_only": [], "section": "5"},
    "main_supplier": {
        "terms": {"train": ["Nhà cung cấp chính", "Nhà cung cấp ruột", "Mối nhập chính", "Mối hàng số một"], "val": ["Nhà cung cấp chủ lực", "Mối nhập quen nhất"]},
        "variants": {"receipts": "nhà cung cấp có **nhiều phiếu nhập nhất** trong kỳ, bỏ phiếu huỷ. Khác nhà cung cấp nhập nhiều tiền nhất",
                     "items": "nhà cung cấp được nhập **nhiều mặt hàng khác nhau nhất** trong kỳ, bỏ phiếu huỷ",
                     "days": "nhà cung cấp có **nhiều ngày có phiếu nhập nhất** trong kỳ, bỏ phiếu huỷ"},
        "val_only": ["days"], "section": "5"},
    # event tables (a section of their own in the rulebook)
    "ev_count": {
        "terms": {"train": ["Lượt, phiếu"], "val": ["Lượt, phiếu"]},
        "variants": {"all": "mỗi dòng của sổ là **một lượt** (một phiếu), kể cả dòng không thu tiền",
                     "paid": "chỉ dòng **có thu tiền** là một lượt (một phiếu); dòng không thu tiền không đếm"},
        "val_only": [], "section": "ev"},
    "ev_repeat": {
        "terms": {"train": ["Khách quay lại", "Khách tới nhiều lần", "Khách đi lại"], "val": ["Khách lặp lại", "Khách trở lại"]},
        "variants": {"n": "xác định khách theo **số điện thoại**; là khách có **từ {n} lượt trở lên** trong kỳ (lượt đếm như dòng Lượt, phiếu)"},
        "val_only": [], "section": "ev"},
    "ev_staff": {
        "terms": {"train": ["Người làm nhiều nhất", "Tay làm chính", "Người gánh việc"], "val": ["Người đứng máy chính", "Người xử lý nhiều nhất"]},
        "variants": {"count": "theo **số lượt** (đếm như dòng Lượt, phiếu), không theo tiền thu", "amount": "theo **tổng tiền thu**, không theo số lượt"},
        "val_only": [], "section": "ev"},
    "ev_free_rate": {
        "terms": {"train": ["Tỉ lệ không thu tiền", "Tỉ lệ làm không công"], "val": ["Tỉ lệ lượt miễn phí"]},
        "variants": {"rows": "số dòng **không thu tiền** chia cho **mọi dòng** của sổ có ngày trong kỳ, tính ra phần trăm"},
        "val_only": [], "section": "ev"},
}
FIXED = {   # rules every synthetic rulebook carries (plain statements, no variants)
    "attr": ("Thuộc tính ghi trong danh mục (xuất xứ, kệ, dòng hàng, mùa bán…)", "lấy theo **cột đó của danh mục hàng**; mặt hàng để trống cột đó thì "
             "**không thuộc giá trị nào**, dù tên hàng có chữ gần giống. Doanh thu, số lượng của một giá trị tính như §1, §3"),
    "lots": ("Lô còn tồn, lô sắp hết hạn", "**lô còn tồn** là lô có số còn lại lớn hơn 0. Tính tại hôm nay D: lô hết hạn trong N ngày tới là lô còn tồn có hạn dùng "
             "từ D đến hết ngày D + N; **lô gần hết hạn nhất** của một mặt hàng là lô còn tồn chưa quá hạn tại D có hạn sớm nhất. **Tiền vốn** nằm ở lô là "
             "số còn lại nhân giá nhập của chính lô đó cho một đơn vị cơ bản"),
    "ev_duration": ("Thời gian xử lý", "số ngày từ **ngày nhận** đến **ngày trả** của phiếu; chỉ tính phiếu đã có ngày trả; phiếu thuộc kỳ theo **ngày nhận**. "
                    "Trung bình làm tròn 2 chữ số thập phân"),
    "ev_period": ("Lượt, phiếu thuộc kỳ nào", "theo **ngày nhận** (ngày hẹn) ghi trên dòng, không theo ngày trả"),
}
MONEY = [300000, 500000, 800000, 1000000, 1500000, 2000000, 3000000, 5000000]

# An attribute written inside the item NAME (train rulebook: "Size M, size L = tên kết thúc bằng (M) hoặc (L)"). Each split draws
# the attribute word, its values and where the name carries it; the rulebook row says how to read it back.
NAME_WORDS = {"train": {"size": ["S", "M", "L", "XL"], "cỡ": ["nhỏ", "vừa", "lớn", "đại"], "loại": ["1", "2", "3"], "số": ["1", "2", "3", "4"]},
              "val": {"hạng": ["A", "B", "C"], "màu": ["đen", "trắng", "đỏ", "xanh"], "dòng": ["Pro", "Plus", "Lite"]}}
NAME_SCHEMES = {
    "paren": ("mặt hàng có tên **kết thúc bằng `(X)`** là hàng {word} X (ví dụ đuôi `({v})`), **không phân biệt hoa thường**; mặt hàng không có đuôi đó "
              "**không thuộc {word} nào**", "{base} ({v})", "%({v})"),
    "word": ("mặt hàng có tên **kết thúc bằng `{word} X`** là hàng {word} X, không phân biệt hoa thường; X phải **đứng riêng sau chữ {word}** "
             "(đuôi `{word} {v}` chỉ tính cho {word} {v}); mặt hàng không có đuôi đó **không thuộc {word} nào**", "{base} {word} {v}", "% {word} {v}"),
    "dash": ("{word} của mặt hàng là phần **sau dấu gạch ngang cuối cùng** của tên (`Tên hàng - X`, ví dụ đuôi `- {v}`), không phân biệt hoa thường; "
             "tên không có đuôi như vậy **không thuộc {word} nào**", "{base} - {v}", "% - {v}"),
    "bracket": ("mặt hàng có tên **bắt đầu bằng `[X]`** là hàng {word} X (ví dụ `[{v}] Tên hàng`), không phân biệt hoa thường; tên không mở đầu bằng ngoặc vuông "
                "**không thuộc {word} nào**", "[{v}] {base}", "[{v}]%"),
    "slash": ("{word} của mặt hàng là phần **sau dấu `/` cuối cùng** của tên (`Tên hàng/X`, ví dụ đuôi `/{v}`), không phân biệt hoa thường; tên không có đuôi "
              "như vậy **không thuộc {word} nào**", "{base}/{v}", "%/{v}"),
}
NAME_VAL_ONLY = ["bracket", "slash"]


def choose_name_rule(rng: np.random.Generator, val: bool = False) -> dict:
    words = NAME_WORDS["val" if val else "train"]
    word = str(rng.choice(sorted(words)))
    values = words[word]
    schemes = NAME_VAL_ONLY if val and rng.random() < 0.7 else [s for s in NAME_SCHEMES if s not in NAME_VAL_ONLY]
    scheme = str(rng.choice(schemes))
    text, name, like = NAME_SCHEMES[scheme]
    term = ", ".join(f"{word} {v}" for v in values)
    return {"term": term[:1].upper() + term[1:], "variant": scheme, "text": text.format(word=word, v=values[1]), "section": "cat",
            "word": word, "values": values, "name": name, "like": like}


# Side books tied to the sales book (round 2): a pay-by-period book (each order paid in periods with due dates) and a book
# of amounts tied to orders by the order id (a deposit, a credit for goods handed in, a delivery fee the shop pays). Each split
# draws the words, the book and the definitions; the rows go to a rulebook section of their own.
PLAN_WORDS = {"train": [("trả góp", "kỳ"), ("trả dần", "đợt"), ("góp theo tháng", "kỳ góp"), ("trả nhiều lần", "lần trả")],
              "val": [("trả chậm", "đợt"), ("góp tháng", "kỳ")]}
LINK_BOOKS = {   # flavor -> book name, the word for its amount, categories of a row, where the row's date sits against the order's
    "deposit": {"book": "sổ đặt cọc", "w": "cọc", "cat": ["Tiền mặt", "Chuyển khoản", "Ví điện tử"], "cat_name": "hình thức cọc"},
    "credit": {"book": "sổ thu đổi hàng cũ", "w": "thu đổi", "cat": ["Còn tốt", "Trầy xước", "Hỏng nhẹ", "Chỉ còn xác"], "cat_name": "tình trạng hàng cũ"},
    "ship": {"book": "sổ giao hàng thuê ngoài", "w": "phí giao", "cat": ["Xe ôm", "Xe tải nhỏ", "Bưu điện", "Nhà xe"], "cat_name": "bên giao"},
    "gift": {"book": "sổ quà tặng kèm đơn", "w": "quà tặng", "cat": ["Tặng phẩm", "Phiếu mua hàng", "Dùng thử"], "cat_name": "loại quà"},
}
LINK_VAL = ["gift"]
LIB_BOOKS = {
    "pl_overdue": {
        "terms": {"train": ["Tiền {p} quá hạn", "Nợ {p} quá hạn", "Khoản {p} trễ hạn"], "val": ["Tiền {p} đọng", "Khoản {p} chậm trả"]},
        "variants": {"strict": "tại hôm nay D: tổng tiền các {p} có hạn **trước ngày D** mà đến hết ngày D chưa trả; {p} có hạn đúng ngày D chưa tính",
                     "incl": "tại hôm nay D: tổng tiền các {p} có hạn **đến hết ngày D** (kể cả hạn đúng ngày D) mà đến hết ngày D chưa trả",
                     "grace": "tại hôm nay D: tổng tiền các {p} đã quá hạn **hơn {n} ngày** (hạn trước ngày D trừ {n} ngày) mà đến hết ngày D chưa trả"},
        "val_only": ["grace"]},
    "pl_late": {
        "terms": {"train": ["Tỉ lệ {p} trả trễ", "Tỉ lệ {p} chậm"], "val": ["Tỉ lệ {p} lỡ hạn"]},
        "variants": {"after": "số {p} **trả sau ngày hạn** chia cho số {p} **đã trả**, chỉ xét các {p} có hạn trong kỳ báo cáo; {p} chưa trả không tính; tính ra phần trăm",
                     "missed": "số {p} **trả sau ngày hạn hoặc đến hết hôm nay chưa trả** chia cho **mọi {p}** có hạn trong kỳ báo cáo; tính ra phần trăm"},
        "val_only": []},
    "lk_period": {
        "terms": {"train": ["Khoản {w} thuộc kỳ nào"], "val": ["Khoản {w} thuộc kỳ nào"]},
        "variants": {"book": "theo **ngày ghi trên dòng của {book}**, không theo ngày của đơn bán",
                     "order": "theo **ngày của đơn bán** mà dòng nối tới, không theo ngày ghi ở {book}"},
        "val_only": []},
    "lk_valid": {
        "terms": {"train": ["Dòng {w} được tính"], "val": ["Dòng {w} được tính"]},
        "variants": {"valid": "chỉ dòng nối với **đơn hợp lệ** (§1: không huỷ, không trả hàng); dòng nối với đơn huỷ, đơn trả hàng thì bỏ",
                     "all": "**mọi dòng** của {book}, kể cả dòng nối với đơn huỷ, đơn trả hàng"},
        "val_only": []},
    "lk_rate": {
        "terms": {"train": ["Tỉ lệ đơn có {w}", "Mức phủ {w}"], "val": ["Độ phủ {w}"]},
        "variants": {"orders": "số **đơn hợp lệ** trong kỳ có dòng ở {book} chia cho số đơn hợp lệ trong kỳ, tính ra phần trăm",
                     "revenue": "**doanh thu** của các đơn hợp lệ trong kỳ có dòng ở {book} chia cho doanh thu kỳ, tính ra phần trăm"},
        "val_only": ["revenue"]},
    "lk_net": {
        "terms": {"train": ["Doanh thu sau {w}", "Thực thu sau {w}", "Doanh thu trừ {w}"], "val": ["Tiền về sau {w}", "Doanh thu ròng sau {w}"]},
        "variants": {"net": "doanh thu kỳ (§1) **trừ** tổng tiền ở {book} của chính các đơn hợp lệ đặt trong kỳ đó (theo ngày đơn bán)"},
        "val_only": []},
}


def choose_books(rng: np.random.Generator, val: bool = False) -> dict:
    plan, p = PLAN_WORDS["val" if val else "train"][int(rng.integers(0, len(PLAN_WORDS["val" if val else "train"])))]
    flavors = LINK_VAL if val and rng.random() < 0.5 else [k for k in LINK_BOOKS if k not in LINK_VAL]
    flavor = str(rng.choice(flavors))
    book = LINK_BOOKS[flavor]
    words = {"plan": plan, "p": p, "flavor": flavor, "book": book["book"], "w": book["w"]}
    out = {"books": {"term": "", "variant": "", "text": "", "section": "", **words}}
    for key, spec in LIB_BOOKS.items():
        variants = list(spec["variants"])
        if not val:
            variants = [v for v in variants if v not in spec["val_only"]] or variants
        elif spec["val_only"] and rng.random() < 0.7:
            variants = spec["val_only"]
        v = str(rng.choice(variants))
        n = int(rng.integers(3, 11))
        fmt = lambda t: t.format(p=p, w=book["w"], book=book["book"], n=n)   # noqa: E731
        term = fmt(str(rng.choice(spec["terms"]["val" if val else "train"])))
        out[key] = {"term": term, "variant": v, "text": fmt(spec["variants"][v]), "section": "bk", "n": n}
    return out


def books_section(rules: dict) -> str:
    w = rules["books"]
    plan, p = w["plan"], w["p"]
    row = lambda k: f"| {rules[k]['term']} | {rules[k]['text']} |\n"   # noqa: E731
    return (f"### Sổ {plan}, {w['book']}\n| Luật | Mặc định |\n|---|---|\n"
            f"| Đơn {plan}, tiền {p} thuộc kỳ nào | đơn {plan} thuộc kỳ theo **ngày mua**; tiền thu của một {p} thuộc kỳ theo **ngày thu**; "
            f"{p} đến hạn trong kỳ tính theo **ngày hạn** |\n"
            f"| Dư nợ {plan} | tại hôm nay D: của các đơn {plan} mua đến hết ngày D, lấy **giá trị đơn trừ trả trước trừ các lần đã thu đến hết ngày D**; "
            f"đơn {plan} **trả xong** tại D là đơn có dư nợ bằng 0 |\n"
            + row("pl_overdue") + row("pl_late") + row("lk_period") + row("lk_valid") + row("lk_rate") + row("lk_net") + "\n")


def choose(rng: np.random.Generator, val: bool = False, names: bool = False, books: bool = False) -> dict:
    """One rule set: for each key a term name, a definition variant and its parameters. names: also an attribute written in
    the item names (drawn last, so the other rules of a seed do not change)."""
    out = {}
    for key, spec in LIB.items():
        variants = list(spec["variants"])
        if not val:
            variants = [v for v in variants if v not in spec["val_only"]]
        elif spec["val_only"] and rng.random() < 0.7:
            variants = spec["val_only"]
        v = str(rng.choice(variants))
        params = {"n": int(rng.integers(2, 6)), "k": int(rng.integers(2, 5)), "x": int(rng.choice(MONEY))}
        text = spec["variants"][v].format(n=params["n"], k=params["k"], x=f"{params['x']:,}".replace(",", "."))
        out[key] = {"term": str(rng.choice(spec["terms"]["val" if val else "train"])), "variant": v, "text": text, "section": spec["section"], **params}
    if names:
        out["variant"] = choose_name_rule(rng, val)
    if books:  # drawn after everything else, so the other rules of a seed do not change
        out.update(choose_books(rng, val))
    return out


def rulebook(train_text: str, rules: dict) -> str:
    """The train rulebook with this split's rules added: rows appended to sections 1, 2, 3 and 5, and one new section for the
    event tables and catalog attributes, placed among the industry sections (before section 7)."""
    text = train_text
    for sec in ("1", "2", "3", "5"):
        rows = "".join(f"| {r['term']} | {r['text']} |\n" for r in rules.values() if r["section"] == sec)
        m = re.search(rf"(?ms)^## {sec}\. .*?(?=^## )", text)
        body = m.group(0).rstrip("\n")
        text = text.replace(m.group(0), body + "\n" + rows + "\n")
    ev = "".join(f"| {r['term']} | {r['text']} |\n" for r in rules.values() if r["section"] == "ev")
    extra = ("### Sổ tiếp nhận, sổ lịch hẹn\n| Luật | Mặc định |\n|---|---|\n" + f"| {FIXED['ev_period'][0]} | {FIXED['ev_period'][1]} |\n" + ev
             + f"| {FIXED['ev_duration'][0]} | {FIXED['ev_duration'][1]} |\n\n"
             "### Danh mục hàng, lô hàng\n| Luật | Mặc định |\n|---|---|\n"
             + f"| {FIXED['attr'][0]} | {FIXED['attr'][1]} |\n| {FIXED['lots'][0]} | {FIXED['lots'][1]} |\n"
             + (f"| {rules['variant']['term']} | {rules['variant']['text']} |\n" if "variant" in rules else "") + "\n")
    if rules.get("variant", {}).get("word") == "size":  # one definition of a size per rulebook: the split's own replaces the train row
        text = re.sub(r"(?m)^\| Size M, size L \|.*\n", "", text)
    if "books" in rules:
        extra += books_section(rules)
    i = text.index("## 7. ")
    return text[:i] + extra + text[i:]
