import itertools

from datamate.core.scoring import score_one
from datamate.qa.agent import _vote_key, decided, vote


def N(x):
    return {"type": "number", "answer": x}


ABS = {"type": "abstain"}


def test_keys_follow_the_scorer():
    assert _vote_key(N(156305)) == _vote_key(N(156305.0))
    assert _vote_key(N(156305)) != _vote_key(N(156304.55))
    assert _vote_key({"type": "percent", "answer": 12.344}) == _vote_key({"type": "percent", "answer": 12.341})
    assert _vote_key({"type": "percent", "answer": 12.34}) != _vote_key({"type": "number", "answer": 12.34})
    assert _vote_key({"type": "string", "answer": "  Cà phê, SỮA "}) == _vote_key({"type": "string", "answer": "cà phê sữa"})
    assert _vote_key({"type": "list", "answer": ["A", "B"]}) != _vote_key({"type": "list", "answer": ["B", "A"]})
    assert _vote_key({"type": "unanswerable", "reason": "x"}) == _vote_key({"type": "unanswerable", "reason": "y"})
    assert _vote_key({"type": "boolean", "answer": True}) != _vote_key(N(1))


def test_table_key_ignores_row_and_column_order():
    a = {"type": "table", "answer": {"columns": ["Tên", "Doanh thu"], "rows": [["An", 10], ["Bình", 20]]}}
    b = {"type": "table", "answer": {"columns": ["doanh thu", "tên"], "rows": [[20, "bình"], [10.0, "an"]]}}
    c = {"type": "table", "answer": {"columns": ["Tên", "Doanh thu"], "rows": [["An", 10], ["Bình", 21]]}}
    assert _vote_key(a) == _vote_key(b) != _vote_key(c)
    ref = {"type": "table", "answer": a["answer"]}
    assert score_one(a, ref) == score_one(b, ref) == 1


def test_vote_threshold_and_tie_goes_to_first():
    assert vote([N(1), N(2), N(1)], 2)[0] == N(1)
    assert vote([N(1), N(2), N(3)], 2)[0] == ABS
    assert vote([N(2), N(1)], 1)[0] == N(2)
    assert vote([ABS, ABS, N(5)], 2)[0] == ABS


def test_decided():
    assert decided([N(1), N(1)], 2, 1)            # K=3: two agree -> third cannot change it
    assert not decided([N(1), N(2)], 2, 1)        # disagree -> need the third
    assert decided([ABS, ABS], 2, 1)              # nobody can reach 2 any more
    assert not decided([N(1), N(1)], 2, 3)        # K=5, need 2: 3 new identical samples could still overtake


def test_early_stop_equals_full_vote():
    """Brute force: stopping when decided() gives the same answer as voting over all K samples."""
    pool = [N(1), N(2), N(3), ABS, {"type": "unanswerable"}]
    for k in (2, 3, 4, 5):
        for need in range(1, k + 1):
            for seq in itertools.product(pool, repeat=k):
                used = []
                for i, p in enumerate(seq):
                    used.append(p)
                    if decided(used, need, k - i - 1):
                        break
                assert vote(used, need)[0] == vote(list(seq), need)[0], (k, need, seq, len(used))
