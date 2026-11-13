"""Several adapters in one run: how the samples of a question are split over them (pipeline.sample with a fake engine)."""
import duckdb
import pytest

from datamate.pipeline import Pipeline, Shop
from datamate.predict import parse_args


class FakeEngine:
    """Answers `SELECT <adapter number>`, so each sample shows which adapter wrote it. Keeps every call's arguments."""

    def __init__(self):
        self.calls = []

    def params(self, **kw):
        return kw

    def n_tokens(self, chat):
        return 10

    def chat(self, chats, params, *, thinking=False, lora=False, prefill=False):
        self.calls.append({"params": params, "lora": lora, "thinking": thinking})
        ps = params if isinstance(params, list) else [params] * len(chats)
        ls = lora if isinstance(lora, list) else [0 if lora else None] * len(chats)
        out = lambda m: type("O", (), {"text": ("<think>\nx\n</think>\n" if thinking else "\n- note\n\n") + f"TYPE: number\n```sql\nSELECT {m + 1}\n```"})()
        return [type("R", (), {"outputs": [out(m) for _ in range(p["n"])]})() for p, m in zip(ps, ls)]


def run(tmp_path, loras, k=8):
    (tmp_path / "rulebook.md").write_text("rules", encoding="utf-8")
    a = parse_args(["--data", str(tmp_path), "--output", str(tmp_path / "out"), "--model", "none", "--k", str(k), "--think-k", "4", "--notes",
                    "--lora-thinking", "--lora", *loras])
    pipe = Pipeline(a, FakeEngine())
    shop = Shop("s", duckdb.connect(), {}, {})
    todo = [{"id": "q_1", "question": "bao nhiêu?"}]
    chats = [[[{"role": "system", "content": s}, {"role": "user", "content": "q"}] for s in pipe.systems]]
    texts, preds, *_ = pipe.sample(shop, todo, chats, lambda: False)
    return pipe, [p["answer"] for p in preds[0]]


def test_one_adapter_keeps_the_single_request_per_prompt(tmp_path):
    pipe, answers = run(tmp_path, ["a"])
    first = pipe.engine.calls[0]
    assert first["lora"] is True and first["params"]["n"] == 4
    assert answers == [1] * 8  # full agreement: no thinking samples
    assert len(pipe.engine.calls) == 1 and "adapter" not in pipe._calls[0][0]


def test_two_adapters_share_each_prompt_and_the_thinking_samples(tmp_path):
    pipe, answers = run(tmp_path, ["a", "b"])
    first, think = pipe.engine.calls[0], pipe.engine.calls[-1]
    assert first["lora"] == [0, 1, 0, 1] and [p["n"] for p in first["params"]] == [2, 2, 2, 2]
    assert answers[:8] == [1, 1, 2, 2, 1, 1, 2, 2]
    assert think["thinking"] and think["lora"] == [0, 1, 0, 1] and answers[8:] == [1, 2, 1, 2]  # a 4-4 split is below --think-below
    assert [pipe.adapter(j) for j in range(12)] == [0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]
    assert [c["adapter"] for c in pipe._calls[0]] == ["a", "a", "b", "b", "a", "a", "b", "b", "a", "b", "a", "b"]


@pytest.mark.parametrize("n, k, want", [(3, 8, [[2, 1, 1], [1, 2, 1]]), (2, 16, [[4, 4], [4, 4]]), (3, 16, [[3, 3, 2], [2, 3, 3]])])
def test_split_gives_the_remainder_to_another_adapter_on_each_prompt(tmp_path, n, k, want):
    (tmp_path / "rulebook.md").write_text("rules", encoding="utf-8")
    a = parse_args(["--data", str(tmp_path), "--output", str(tmp_path), "--model", "none", "--k", str(k), "--lora", *"abc"[:n]])
    pipe = Pipeline(a)
    assert [pipe.split(0), pipe.split(1)] == want


def test_gift_is_not_a_rise():
    from datamate.sqlexec import asks_change
    assert asks_change("Doanh thu tháng 5 tăng bao nhiêu % so với tháng 4?")
    assert asks_change("dthu thang 5 tang hay giam may % so thang 4")
    assert not asks_change("Độ phủ quà tặng tháng 6/2025 là bao nhiêu %?")
    assert not asks_change("don co qua tang chiem bao nhieu % so don thang 6")
    assert not asks_change("Hàng tặng kèm chiếm mấy % số lượng bán tháng này?")
    assert not asks_change("Kệ tầng 2 chiếm bao nhiêu % doanh thu?")


def test_watchdog_interrupts_a_runaway_statement():
    import time
    import duckdb
    import pytest
    from datamate import views
    db = duckdb.connect()
    t0 = time.time()
    with pytest.raises(duckdb.Error):
        with views.watchdog(db, 1):
            db.execute("SELECT count(*) FROM range(3000000) a, range(3000000) b WHERE a.range + b.range < 0").fetchone()
    assert time.time() - t0 < 20
    assert db.execute("SELECT 42").fetchone() == (42,)


def test_qty_redone_from_recorded_quantity():
    """A mapping that multiplied every line by the catalog factor (base-unit lines too) is redone from qty_recorded / unit_recorded."""
    import duckdb
    from datamate import views
    db = duckdb.connect()
    rows = []
    for d in range(1, 29):
        for j in range(3):
            rows.append((f"A{d}-{j}", f"2025-01-{d:02d}", "Gạch", 2, "viên", 2 * 1300))
        rows.append((f"B{d}", f"2025-01-{d:02d}", "Gạch", 1, "thiên", 1300000))
    db.execute("CREATE TABLE s (oid VARCHAR, d DATE, name VARCHAR, q DOUBLE, u VARCHAR, amt DOUBLE)")
    db.executemany("INSERT INTO s VALUES (?, ?, ?, ?, ?, ?)", rows)
    bad = """SELECT oid AS order_id, d AS order_date, NULL AS order_ts, NULL AS product_code, NULL AS product_group, name AS product_name,
        q * 1000 AS qty, 'viên' AS unit, amt AS amount, NULL AS customer_id, NULL AS customer_name, NULL AS staff, NULL AS channel, NULL AS payment,
        false AS is_cancelled, false AS is_returned, 'item' AS line_kind, q AS qty_recorded, u AS unit_recorded FROM s"""
    views.build(db, bad)
    assert db.execute("SELECT sum(qty) FROM v_sales").fetchone()[0] == 28 * (3 * 2000 + 1000)
    views.build(db, views.qty_by_price(bad))
    assert db.execute("SELECT sum(qty) FROM v_sales").fetchone()[0] == 28 * (3 * 2 + 1000)
    assert db.execute("SELECT count(DISTINCT unit) FROM v_sales").fetchone()[0] == 1
