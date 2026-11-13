"""Mechanisms that replace hardcoded lists: LLM label classification, rulebook-anchored helpers, period phrasing.

The phrasings below are written for these tests (not copied from any question set): they cover ways of writing a
period that a new question set may use.
"""
import json

import pandas as pd
import pytest

from datamate.qa import helpers, qnorm, qparse, rule_anchors
from datamate.qa.agent import helpers_doc
from datamate.semantic import labels

AS_OF = pd.Timestamp("2026-05-20")


def periods(q):
    return [p[1:] for p in qparse.resolve_periods(qnorm.normalize(q), AS_OF)]


@pytest.mark.parametrize("q, want", [
    ("dthu th3/26 bnhiu", [("2026-03-01", "2026-03-31")]),
    ("Doanh thu T12/2025?", [("2025-12-01", "2025-12-31")]),
    ("doanh thu thg 7/2025", [("2025-07-01", "2025-07-31")]),
    ("doanh thu 7/25", [("2025-07-01", "2025-07-31")]),
    ("doanh thu 07-2025 là bao nhiêu", [("2025-07-01", "2025-07-31")]),
    ("dthu thang 2-26", [("2026-02-01", "2026-02-28")]),
    ("Th 11/25 doanh thu", [("2025-11-01", "2025-11-30")]),
    ("Dthu tháng 3 năm 2026", [("2026-03-01", "2026-03-31")]),
    ("doanh thu tháng giêng", [("2026-01-01", "2026-01-31")]),
    ("tháng tư năm nay bán bao nhiêu", [("2026-04-01", "2026-04-30")]),
    ("tháng 6 năm ngoái bán bao nhiêu", [("2025-06-01", "2025-06-30")]),
    ("doanh thu q2/2025", [("2025-04-01", "2025-06-30")]),
    ("doanh thu Q4.2025", [("2025-10-01", "2025-12-31")]),
    ("doanh thu quý 1 năm 2026", [("2026-01-01", "2026-03-31")]),
    ("dthu tuan truoc", [("2026-05-11", "2026-05-17")]),
    ("Doanh thu 30 ngày qua", [("2026-04-21", "2026-05-20")]),
    ("từ 1/3 đến 15/3 dthu", [("2026-03-01", "2026-03-15")]),
    ("Doanh thu năm ngoái", [("2025-01-01", "2025-12-31")]),
])
def test_period_phrasings(q, want):
    assert periods(q) == want


# ---------------------------------------------------------------- label classification

class FakeLLM:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def chat(self, messages, **kw):
        self.prompts.append(messages[-1]["content"])
        return type("R", (), {"content": json.dumps(self.reply, ensure_ascii=False)})()


def _orders():
    return pd.DataFrame({
        "order_id": ["1", "2", "3", "4"],
        "channel": ["Tại tiệm", "Temu", "Zalo", "Temu"],
        "status_raw": ["Xong", "Khách bom hàng", "Xong", "Đã hủy"],
        "status": ["completed", None, "completed", "cancelled"],
        "customer_name": ["Khách qua đường", "Nguyễn An", "Khách qua đường", "Lê Bình"],
    })


def test_classify_and_apply():
    llm = FakeLLM({"channel": {"Tại tiệm": "store", "Temu": "marketplace", "Zalo": "chat", "Ghost": "store"},
                   "status": {"Xong": "completed", "Khách bom hàng": "cancelled", "Đã hủy": "completed"},
                   "walk_in": ["Khách qua đường", "Người lạ"]})
    lab = labels.classify(llm, {"orders": _orders()}, "# Shop\nBán điện thoại.", "")
    assert lab["channel"] == {"Tại tiệm": "store", "Temu": "marketplace", "Zalo": "chat"}  # unknown keys dropped
    assert lab["walk_in"] == ["Khách qua đường"]                                            # only asked values
    o = labels.apply_labels({"orders": _orders()}, {"labels": lab})["orders"]
    assert o["channel_type"].tolist() == ["store", "marketplace", "chat", "marketplace"]
    assert o["status"].tolist() == ["completed", "cancelled", "completed", "cancelled"]      # keyword 'hủy' keeps
    assert o["customer_name"].tolist() == [None, "Nguyễn An", None, "Lê Bình"]


def test_apply_without_llm_labels_uses_fallback():
    o = labels.apply_labels({"orders": _orders().assign(channel=["store", "Sàn TMĐT", "zalo_oa", "marketplace"])},
                            {})["orders"]
    assert o["channel_type"].tolist() == ["store", "marketplace", None, "marketplace"]


def test_debts_exclude_marketplace_by_channel_type():
    o = pd.DataFrame({"order_id": ["a", "b"], "order_date": pd.to_datetime(["2026-05-01"] * 2),
                      "status": ["completed"] * 2, "channel": ["Quầy", "Temu"],
                      "channel_type": ["store", "marketplace"], "payment_method": ["debt", "debt"],
                      "total_payable": [100.0, 200.0], "debt_remaining": [100.0, 200.0],
                      "customer_name": ["A", "B"]})
    d = helpers.debts({"orders": o}, "2026-05-10")
    assert d["order_id"].tolist() == ["a"]


# ---------------------------------------------------------------- rulebook anchors

BOOK = """## 4. Công nợ
| Luật | Mặc định |
|---|---|
| Đơn được tính vào nợ | đơn không huỷ; **đơn trả hàng vẫn tính** nếu còn nợ |
| Nợ một đơn tại ngày X | phần chưa trả của **tổng đơn** |
| Đơn COD, đơn bán qua sàn TMĐT | **không phải khách nợ** |
| Tuổi nợ | số ngày tính từ **ngày đặt** |
## 6. Theo ngành
| Luật | Mặc định |
|---|---|
| Trả góp | Đơn trả góp là đơn **ghi nợ** |
"""


def test_anchor_status_and_flags():
    rules = qparse.parse_rulebook(BOOK)
    st = rule_anchors.status(rules)
    assert st["debts"] == "ok" and st["qty_in"] == "absent"
    assert rule_anchors.flags(rules) == {"installment_is_credit": True, "delivering_counts": True}
    changed = qparse.parse_rulebook(BOOK.replace("**đơn trả hàng vẫn tính**", "đơn trả hàng không tính"))
    assert rule_anchors.status(changed)["debts"] == "changed"
    assert rule_anchors.flags(qparse.parse_rulebook(BOOK.replace("| Trả góp |", "| Khác |")
                                                    .replace("ghi nợ", "bán thường")))["installment_is_credit"] is False


def test_withdrawn_helper_leaves_the_doc():
    doc = helpers_doc({"debts"})
    assert "- debts(at)" not in doc and "- debt_flow(start, end)" in doc and "- qty_in(" in doc


def test_hints_follow_anchor_status():
    q = "đến hết hôm nay khách còn nợ bao nhiêu"
    tables = {"orders": pd.DataFrame({"order_id": ["1"]})}
    ok = qparse.analyze(q, AS_OF, tables, helper_rules={"debts": ("ok", "Tuổi nợ: từ ngày đặt")})
    assert "debts(" in ok and "[rulebook: Tuổi nợ" in ok
    gone = qparse.analyze(q, AS_OF, tables, helper_rules={"debts": ("changed", "")})
    assert "debts(" not in gone


# ---------------------------------------------------------------- product attribute refill

from datamate.semantic import attrs  # noqa: E402


class FakeStore:
    def __init__(self, sheets):
        self.sheets = sheets

    def list_files(self):
        return sorted({f for f, _ in self.sheets})

    def sheet_names(self, f):
        return [s for ff, s in self.sheets if ff == f]

    def read_table(self, f, s):
        return self.sheets[(f, s)].copy()


def _catalogue():
    return FakeStore({("DanhMuc.xlsx", "Sheet1"): pd.DataFrame({
        "Mã": ["P1", "P2", "P3"], "Hiệu": ["Acme", "Beta", None], "Bán?": ["có", "ngừng", "có"],
        "Giá gốc": ["10.000", "20.000", "5,5k"]})})


def test_catalogue_found_by_key_values_not_header_names():
    p = pd.DataFrame({"product_id": ["P1", "P2", "P3"], "name": ["A", "B", "C"]})
    (f, sh, df, key), = attrs.catalogue_sheets(_catalogue(), p)
    assert (f, sh) == ("DanhMuc.xlsx", "Sheet1") and key.tolist() == ["P1", "P2", "P3"] and "Mã" not in df


def test_enrich_from_llm_sources():
    p = pd.DataFrame({"product_id": ["P1", "P2", "P3"], "name": ["A", "B", "C"], "brand": [None] * 3})
    llm = FakeLLM({"brand": {"file": "DanhMuc.xlsx", "sheet": "Sheet1", "header": "Hiệu"},
                   "active": {"file": "DanhMuc.xlsx", "sheet": "Sheet1", "header": "Bán?", "true_values": ["có"]},
                   "cost_price": {"file": "DanhMuc.xlsx", "sheet": "Sheet1", "header": "Giá gốc"},
                   "size": {"file": "DanhMuc.xlsx", "sheet": "Sheet1", "header": "Không có cột này"}})
    t = {"products": p}
    prop = attrs.propose(llm, t, _catalogue())
    assert sorted(prop["map"]) == ["active", "brand", "cost_price"]          # invented header rejected
    meta = {}
    attrs.enrich(t, _catalogue(), meta, prop["map"])
    out = t["products"]
    assert out["brand"].tolist()[:2] == ["Acme", "Beta"] and pd.isna(out["brand"].iloc[2])
    assert out["active"].tolist() == [True, False, True]
    assert out["cost_price"].tolist() == [10000.0, 20000.0, 5500.0]
    assert len(meta["fixes"]) == 3


def test_enrich_keeps_filled_attributes():
    p = pd.DataFrame({"product_id": ["P1", "P2", "P3"], "name": ["A", "B", "C"], "brand": ["X", "Y", "Z"]})
    t = {"products": p}
    attrs.enrich(t, _catalogue(), {}, {"brand": {"file": "DanhMuc.xlsx", "sheet": "Sheet1", "header": "Hiệu"}})
    assert t["products"]["brand"].tolist() == ["X", "Y", "Z"]


def test_columns_already_in_the_layer_are_not_offered_again():
    store = FakeStore({("Thuoc.xlsx", "S"): pd.DataFrame({
        "Mã": ["P1", "P2", "P3", "P4"], "Hoạt chất": ["Para", "Ibu", "Para", "Vit C"],
        "Kê đơn": ["x", "", "x", ""], "Hãng": ["DHG", "Sanofi", "DHG", "Traphaco"]})})
    p = pd.DataFrame({"product_id": ["P1", "P2", "P3", "P4"], "name": list("ABCD"),
                      "active_ingredient": ["Para", "Ibu", "Para", "Vit C"],
                      "is_prescription": [True, False, True, False]})
    llm = FakeLLM({})
    attrs.propose(llm, {"products": p}, store)
    listed = llm.prompts[0].split("Columns of the shop's raw catalogue sheets")[1]
    assert "'Hãng'" in listed and "'Hoạt chất'" not in listed and "'Kê đơn'" not in listed


# ---------------------------------------------------------------- structural total rows, data-derived units

from datamate.semantic.validate import _total_rows  # noqa: E402


def test_order_total_row_is_dropped_whatever_its_label():
    ln = pd.DataFrame({"order_id": ["A", "A", "A", "B", "B"],
                       "product_name": ["Trà", "Bánh", "Σ đơn A", "Cà phê", "Phí ship"],
                       "qty": [1, 2, None, 1, None], "unit_price": [10, 5, None, 20, None],
                       "line_amount": [10.0, 10.0, 20.0, 20.0, 15.0]})
    assert _total_rows(ln, "order_lines").tolist() == [False, False, True, False, False]


def test_units_come_from_the_shop_data():
    tables = {"order_lines": pd.DataFrame({"unit": ["nồi", "cặp đôi", None]}),
              "products": pd.DataFrame({"base_unit": ["suất"]})}
    rx = qparse.units_rx(tables).split("|")
    assert {"noi", "cap", "suat", "kg"} <= set(rx)
    hints = qparse.helper_hints("tháng này bán được bao nhiêu suất", {**tables, "orders": pd.DataFrame()})
    assert any("qty_in(sales(start, end), 'suat')" in h for h in hints)
