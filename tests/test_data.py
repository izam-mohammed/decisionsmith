import json

import pytest

from decisionsmith.schema import compile_schema
from decisionsmith.training.data import DataError, Row, load, split
from tests.conftest import Ticket


def test_csv(toy_csv):
    rows = load(toy_csv, Ticket, group_by="group")
    assert len(rows) == 60 and rows[0].id == "r0" and rows[0].group == "0"
    assert rows[0].targets["team"] == [1.0, 0.0, 0.0] and rows[0].targets["wants_refund"] == [1.0, 0.0]
    assert set(rows[0].questions) == {"team", "wants_refund"}


def test_csv_blank_cells_and_errors(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("text,team,wants_refund\nhello,billing,\nbye,,yes\nnothing,,\n")
    rows = load(str(p), compile_schema(Ticket))
    assert [set(r.targets) for r in rows] == [{"team"}, {"wants_refund"}]
    p.write_text("body,team\nhello,billing\n")
    with pytest.raises(DataError, match="'text' column"):
        load(str(p), Ticket)
    p.write_text("text,team\nhello,legal\n")
    with pytest.raises(DataError, match="d.csv line 2 field 'team'"):
        load(str(p), Ticket)


def test_answers_jsonl(tmp_path):
    p = tmp_path / "a.jsonl"
    recs = [
        {"id": "a", "text": "x", "answers": {"team": {"billing": 3, "sales": 1}, "wants_refund": True}},
        {"text": "y", "answers": {"team": "sales", "wants_refund": None}},
    ]
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n\n")
    rows = load(str(p), Ticket)
    assert rows[0].targets["team"] == [0.75, 0.0, 0.25] and rows[0].targets["wants_refund"] == [0.0, 1.0]
    assert rows[1].id == "row1" and set(rows[1].targets) == {"team"}


@pytest.mark.parametrize(
    "rec,match",
    [
        ({"text": "x", "answers": {"nope": 1}}, "unknown fields"),
        ({"text": "x", "answers": {"team": {"legal": 1}}}, "not options"),
        ({"text": "x", "answers": {"team": {"billing": -1}}}, "non-negative"),
        ({"text": "x", "answers": {"team": {"billing": 0}}}, "empty distribution"),
        ({"text": "", "answers": {"team": "sales"}}, "missing 'text'"),
        ({"text": "x", "answers": []}, "must be an object"),
        ("not a dict", "expected an object"),
    ],
)
def test_answer_errors(rec, match):
    with pytest.raises(DataError, match=match):
        load([rec], Ticket)


def test_typed_decisions():
    rec = {
        "id": "c1",
        "state": json.dumps({"email": "refund me"}),
        "questions": json.dumps(
            {
                "q1": {"type": "choice", "instructions": "?", "criteria": {"a": "", "b": ""}},
                "q2": {"type": "noul", "instructions": "?"},
                "q3": {"type": "score", "instructions": "?", "criteria": ["lo", "hi"]},
            }
        ),
        "gold": json.dumps(
            {
                "q1": {"probabilities": {"a": 0.2, "b": 0.6}},
                "q2": {"label": "True"},
                "q3": {"probabilities": {"1": 1.0}},
            }
        ),
    }
    (r,) = load([rec])
    assert r.text == {"email": "refund me"} and r.targets["q1"] == pytest.approx([0.25, 0.75])
    assert r.targets["q2"] == [0.0, 1.0] and r.targets["q3"] == [0.0, 1.0]
    assert load([{**rec, "state": "{not json"}])[0].text == "{not json"
    assert load([{**rec, "state": None}])[0].text == ""


@pytest.mark.parametrize(
    "patch,match",
    [
        ({"questions": "{bad"}, "not valid JSON"),
        ({"gold": []}, "need 'questions' and 'gold'"),
        ({"gold": {"q9": {"label": "a"}}}, "unknown or missing"),
        ({"gold": {"q1": {"label": "zzz"}}}, "needs 'probabilities'"),
        (
            {
                "questions": {"q1": {"type": "choice", "instructions": "?", "criteria": {}}},
                "gold": {"q1": {"label": "a"}},
            },
            "no options",
        ),
    ],
)
def test_typed_errors(patch, match):
    base = {
        "state": "s",
        "questions": {"q1": {"type": "choice", "instructions": "?", "criteria": ["a", "b"]}},
        "gold": {"q1": {"label": "a"}},
    }
    with pytest.raises(DataError, match=match):
        load([{**base, **patch}])


def test_files_and_ids(tmp_path):
    with pytest.raises(DataError, match="no such file"):
        load(str(tmp_path / "nope.csv"), Ticket)
    bad = tmp_path / "b.jsonl"
    bad.write_text("{oops\n")
    with pytest.raises(DataError, match="line 1: not valid JSON"):
        load(str(bad), Ticket)
    with pytest.raises(DataError, match="needs a schema"):
        load([{"text": "x", "answers": {"team": "sales"}}])
    assert len(load([{"id": "a", "text": "x", "answers": {"team": "sales"}}] * 2, Ticket)) == 1
    with pytest.raises(DataError, match="appears twice with different text or labels"):
        load(
            [
                {"id": "a", "text": "x", "answers": {"team": "sales"}},
                {"id": "a", "text": "y", "answers": {"team": "billing"}},
            ],
            Ticket,
        )


def _rows(n, group=None):
    return [Row("r%d" % i, "t", {}, {"q": [1.0]}, group(i) if group else None) for i in range(n)]


def test_split():
    tr, ca, te = split(_rows(100), seed=1)
    assert (len(tr), len(ca), len(te)) == (75, 10, 15)
    assert split(_rows(100), seed=1)[2][0].id == te[0].id
    assert not {r.id for r in tr} & {r.id for r in te}
    tr, ca, te = split(_rows(100, group=lambda i: str(i // 10)))
    assert all(
        len({r.group for r in part} & {r.group for r in other}) == 0 for part, other in ((tr, te), (tr, ca), (ca, te))
    )
    assert len(split(_rows(10_000))[1]) == 400
    with pytest.raises(DataError, match="at least 20"):
        split(_rows(19))
    with pytest.raises(DataError, match="groups are too large"):
        split(_rows(40, group=lambda i: str(i // 20)))
