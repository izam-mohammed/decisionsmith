"""Flow 1b: a coding agent labels, trains and evaluates through the MCP tools (no LLM API key)."""

import csv
import json
import os

import pytest

import decisionsmith as ds
from decisionsmith import cli, golden_session
from decisionsmith import mcp as tools
from decisionsmith.checks import check
from decisionsmith.log import Log
from decisionsmith.schema import compile_schema
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, corpus, truth

LABELS = ["billing", "technical", "sales"]
SCHEMA = "tests/conftest.py:Ticket"


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def start(tmp_path, n=30, schema=Ticket, **kw):
    out = str(tmp_path / "golden.csv")
    rows = ds.golden(corpus(n), "agent", n, "random", schema=schema, out=out, verbose=False, **kw)
    return golden_session.session_path(out), rows


def answer(text, fields=("team", "wants_refund")):
    t = truth(text)
    return {f: t[f] for f in fields}


def second_ids(session):
    return {r["check_id"] for r in golden_session.load(session)["rows"] if r.get("check_id")}


def label_all(session, wrong=(), agent="claude-code"):
    """A scripted agent: label every batch; on the second answer, answer the texts in `wrong` differently."""
    while True:
        b = tools.golden_batch(session, 7)
        if not b["items"]:
            return b
        answers, second = [], second_ids(session)
        for item in b["items"]:
            got = answer(item["text"], item.get("fields", list(b["fields"])))
            if item["id"] in second and item["text"] in wrong:
                got["team"] = next(x for x in LABELS if x != got["team"])
            answers.append({"id": item["id"], "answers": got})
        res = tools.golden_submit(session, answers, agent)
        assert res["accepted"] == len(answers) and not res["rejected"]


# the session


def test_agent_session_round_trip(tmp_path, capsys):
    out = str(tmp_path / "golden.csv")
    rows = ds.golden(corpus(30), "agent:claude-code", 20, "random", schema=Ticket, out=out)
    session = golden_session.session_path(out)
    assert session.endswith("golden.session.json") and os.path.exists(session) and not os.path.exists(out)
    assert "golden: 20 rows (random) for a coding agent" in capsys.readouterr().out
    assert len(rows) == 20 and all(r["labelled_by"] is None and r["answers"] == {} for r in rows)
    assert {r["split"] for r in rows} == {"train", "test"}
    data = golden_session.load(session)
    assert data["agent"] == "claude-code" and data["labels"] is None and data["out"] == "golden.csv"
    assert golden_session.schema_of(data).questions() == compile_schema(Ticket).questions()
    b = tools.golden_batch(session, 5)
    assert "pass" not in b and len(b["items"]) == 5 and b["remaining"] == 20
    assert set(b["items"][0]) == {"id", "text"} and b["fields"]["team"]["options"] == LABELS
    assert b["fields"]["team"]["descriptions"]["billing"] == "payments and refunds" and "skip" in b["instructions"]
    done = label_all(session, agent=None)
    assert done["instructions"] == "" and "golden_finish" in done["next"]
    st = tools.golden_status(session)
    assert st["labelled"] == 20 and st["rechecked"] >= 1 and st["agreement"] == 1.0 and st["disagreed"] == 0
    assert st["labelled_by"] == {"agent:claude-code": 20} and sum(st["balance"]["team"].values()) == 20
    assert st["split"]["train"] + st["split"]["test"] == 20
    res = tools.golden_finish(session)
    got = read(out)
    assert res["rows"] == 20 and len(got) == 20 and res["path"] == os.path.abspath(out)
    assert list(got[0]) == ["id", "text", "team", "wants_refund", "split", "labelled_by", "checked"]
    assert {r["checked"] for r in got} == {"", "agreed"} and all(r["team"] == truth(r["text"])["team"] for r in got)
    with pytest.raises(FileExistsError):
        tools.golden_finish(session)
    assert tools.golden_finish(session, str(tmp_path / "g.jsonl"))["rows"] == 20
    st = tools.golden_status(session)
    assert st["finished"] and st["next"] == "finished: golden.csv is written; edit it there"
    for call in (
        lambda: tools.golden_batch(session),
        lambda: tools.golden_submit(session, [{"id": got[0]["id"], "answers": answer(got[0]["text"])}]),
        lambda: tools.golden_add(
            session, [{"text": "a new text after finishing", "answers": {"team": "billing", "wants_refund": True}}]
        ),
    ):
        with pytest.raises(ValueError, match="is finished .*golden.csv was written"):
            call()


def test_labels_session_and_rechecks_flag_disagreements(tmp_path):
    session, _ = start(tmp_path, 40, schema=LABELS)
    data = golden_session.load(session)
    assert data["labels"] == LABELS and golden_session.spec_of(data) == LABELS
    rechecked = {r["text"]: r["id"] for r in data["rows"] if r["recheck"]}
    b = tools.golden_batch(session, 100)
    first = [{"id": i["id"], "answers": {"label": truth(i["text"])["team"]}} for i in b["items"]]
    tools.golden_submit(session, first)
    b2 = tools.golden_batch(session, 100)
    assert {i["text"] for i in b2["items"]} == set(rechecked) and "second answer" in b2["instructions"]
    assert not {i["id"] for i in b2["items"]} & set(rechecked.values())
    wrong, unsure = sorted(rechecked)[:2]
    second = []
    for i in b2["items"]:
        team = truth(i["text"])["team"]
        if i["text"] == wrong:
            second.append({"id": i["id"], "answers": {"label": next(x for x in LABELS if x != team)}})
        elif i["text"] == unsure:
            second.append({"id": i["id"], "skip": " two could fit "})
        else:
            second.append({"id": i["id"], "answers": {"label": team}})
    assert tools.golden_submit(session, second)["disagreed"] == 2
    st = tools.golden_status(session)
    assert {d["id"] for d in st["disagreements"]} == {rechecked[wrong], rechecked[unsure]}
    assert any(d["second"] == "skipped: two could fit" for d in st["disagreements"])
    assert st["agreement"] == (len(rechecked) - 2) / len(rechecked)
    res = tools.golden_finish(session)
    assert res["disagreed"] == 2 and "left blank for you" in res["message"]
    by_text = {r["text"]: r for r in read(tmp_path / "golden.csv")}
    assert by_text[wrong]["label"] == "" and by_text[wrong]["checked"] == "disagreed: label"
    assert by_text[unsure]["label"] == "" and by_text[unsure]["labelled_by"] == "agent:coding-agent"


def test_submit_rejects_bad_items_one_by_one(tmp_path):
    session, _ = start(tmp_path, 6)
    ids = [i["id"] for i in tools.golden_batch(session)["items"]]
    res = tools.golden_submit(
        session,
        [
            "not an object",
            {"id": "nope", "answers": {}},
            {"id": ids[0], "answers": "billing"},
            {"id": ids[0], "answers": {"team": "billing", "colour": "red"}},
            {"id": ids[0], "answers": {"team": "billing"}},
            {"id": ids[0], "answers": {"team": "legal", "wants_refund": True}},
            {"id": ids[0], "answers": {"team": "billing", "wants_refund": "maybe"}},
            {"id": ids[1], "answers": {"team": "billing", "wants_refund": "yes"}},
            {"id": ids[1], "answers": {"team": "sales", "wants_refund": "no"}},
            {"id": ids[2], "skip": "two teams could fit"},
        ],
    )
    errors = [r["error"] for r in res["rejected"]]
    assert res["accepted"] == 2 and len(errors) == 8
    for part in ("item 0", "no row with id 'nope'", "must be an object", "unknown fields ['colour']", "missing"):
        assert any(part in e for e in errors), part
    assert any("'legal' is not an option" in e for e in errors) and any("'maybe'" in e for e in errors)
    assert "already answered" in errors[-1]
    st = tools.golden_status(session)
    assert st["skipped"] == 1 and st["skipped_rows"][0]["why"] == "two teams could fit" and st["labelled"] == 1
    with pytest.raises(ValueError, match="must be a list"):
        tools.golden_submit(session, {"id": ids[0]})
    with pytest.raises(ValueError, match="agent name"):
        tools.golden_submit(session, [], agent="bad name!")
    with pytest.raises(ValueError, match="size must be at least 1"):
        tools.golden_batch(session, 0)


def test_synthetic_examples_count_only_after_an_agreeing_recheck(tmp_path):
    session, _ = start(tmp_path, 4)
    label_all(session)
    written = {
        "you charged me twice again today": {"team": "billing", "wants_refund": False},
        "the export button crashes": {"team": "technical", "wants_refund": False},
        "what does the team plan cost": {"team": "sales", "wants_refund": False},
    }
    examples = [{"text": t, "answers": a} for t, a in written.items()]
    bad = [
        {"text": corpus(4)[0], "answers": answer(corpus(4)[0])},
        {"text": " ", "answers": {}},
        {"text": "x y z", "answers": {"team": "hr", "wants_refund": False}},
        "not an object",
    ]
    res = tools.golden_add(session, examples + bad, agent="claude-code")
    assert res["accepted"] == 3 and len(res["rejected"]) == 4 and res["synthetic"] == 3
    assert "already in the session" in res["rejected"][0]["error"]
    with pytest.raises(ValueError, match="must be a list"):
        tools.golden_add(session, {"text": "x"})
    b = tools.golden_batch(session, 10)
    assert {i["text"] for i in b["items"]} == set(written) and not {i["id"] for i in b["items"]} & set(
        rows_ids(session)
    )
    agree, disagree, _ = b["items"]
    tools.golden_submit(
        session,
        [
            {"id": agree["id"], "answers": written[agree["text"]]},
            {"id": disagree["id"], "answers": {"team": "billing", "wants_refund": True}},
        ],
    )
    res = tools.golden_finish(session)
    synthetic = [r for r in read(tmp_path / "golden.csv") if r["labelled_by"] == "agent:claude-code:synthetic"]
    assert [r["text"] for r in synthetic] == [agree["text"]] and synthetic[0]["split"] == "train"
    assert res["dropped_synthetic"] == 2 and "2 synthetic rows left out" in res["message"]
    assert tools.golden_status(session)["disagreements"][0]["synthetic"] is True


def rows_ids(session):
    return [r["id"] for r in golden_session.load(session)["rows"]]


def test_human_labels_from_the_log_are_kept(tmp_path):
    db = str(tmp_path / "decisions.db")
    student = FakeEngine(truth, confidence=0.6, name="student")
    with ds.harness(Ticket, teacher=FakeEngine(truth), student=student, mode="shadow", log=db, collect=1.0) as h:
        results = [h.decide(t) for t in corpus(6)]
        h.label(results[0].id, team="billing", wants_refund=False)
        h.label(results[1].id, team="technical")
    with Log(db) as log:
        texts = {r["id"]: r["text"] for r in log.rows("Ticket")}
    out = str(tmp_path / "g.csv")
    ds.golden(db, "agent", 6, "uncertain", schema=Ticket, out=out, verbose=False)
    session = golden_session.session_path(out)
    b = tools.golden_batch(session, 10)
    assert len(b["items"]) == 5
    partial = next(i for i in b["items"] if i["text"] == texts[results[1].id])
    assert partial["fields"] == ["wants_refund"]
    tools.golden_submit(
        session,
        [{"id": i["id"], "answers": answer(i["text"], i.get("fields", ["team", "wants_refund"]))} for i in b["items"]],
    )
    by_text = {r["text"]: r for r in read(tools.golden_finish(session)["path"])}
    assert by_text[texts[results[0].id]]["labelled_by"] == "human"
    assert by_text[texts[results[1].id]]["labelled_by"] == "agent:coding-agent+human"
    assert by_text[texts[results[1].id]]["team"] == "technical"


def test_session_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="no labelling session"):
        golden_session.load(str(tmp_path / "none.json"))
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    with pytest.raises(ValueError, match="not valid JSON"):
        golden_session.load(str(bad))
    bad.write_text("[]")
    with pytest.raises(ValueError, match="not a decisionsmith labelling session"):
        golden_session.load(str(bad))
    with pytest.raises(ValueError, match="give out="):
        ds.golden(corpus(5), "agent", 5, "random", schema=LABELS, out=None)
    with pytest.raises(ValueError, match="agent name"):
        ds.golden(corpus(5), "agent:no spaces", 5, "random", schema=LABELS)
    with pytest.raises(ValueError, match="must end in .csv"):
        ds.golden(corpus(5), "agent", 5, "random", schema=LABELS, out=str(tmp_path / "g.txt"))
    session, _ = start(tmp_path, 5)
    with pytest.raises(FileExistsError):
        start(tmp_path, 5)
    assert start(tmp_path, 5, overwrite=True)[0] == session
    with pytest.raises(ValueError, match="nothing labelled yet"):
        tools.golden_finish(session)
    assert golden_session.agent_of("claude-opus-5") is None and golden_session.agent_of("agent") == ""
    only_human = golden_session.start(
        [("a b", {}, "train")],
        compile_schema(Ticket),
        str(tmp_path / "h.csv"),
        agent=None,
        strategy="random",
        overwrite=False,
        verbose=False,
    )
    assert only_human[0]["labelled_by"] is None


def test_every_row_gets_rechecked_at_least_once():
    rows = [{"text": "t%d" % i, "labelled_by": None, "recheck": False} for i in range(3)]
    golden_session._pick_rechecks(rows, lambda text: 0.9)
    assert sum(r["recheck"] for r in rows) == 1
    golden_session._pick_rechecks([], lambda text: 0.9)


# the second answer can only come from an id golden_batch handed out


def first_pass(session):
    b = tools.golden_batch(session, 100)
    res = tools.golden_submit(session, [{"id": i["id"], "answers": answer(i["text"])} for i in b["items"]])
    assert res["accepted"] == len(b["items"]) and res["to_label"] == 0
    return {i["text"]: i["id"] for i in b["items"]}


def recheck_rows(session):
    return [r for r in golden_session.load(session)["rows"] if r["recheck"]]


def test_a_duplicate_in_the_same_call_is_not_a_second_answer(tmp_path):
    session, _ = start(tmp_path, 20)
    rows = golden_session.load(session)["rows"]
    rc = next(r for r in rows if r["recheck"])
    plain = next(r for r in rows if not r["recheck"])
    items = [{"id": r["id"], "answers": answer(r["text"])} for r in (rc, rc, plain, plain)]
    res = tools.golden_submit(session, items)
    assert res["accepted"] == 2 and res["rechecked"] == 0 and res["agreed"] == 0
    assert [e["error"] for e in res["rejected"]] == [
        "already answered; to change a label, edit golden.csv after --finish"
    ] * 2


def test_a_retry_after_a_timeout_is_rejected_and_batches_are_stable(tmp_path):
    session, _ = start(tmp_path, 20)
    ids = first_pass(session)
    again = tools.golden_submit(session, [{"id": i, "answers": answer(t)} for t, i in ids.items()])
    assert again["accepted"] == 0 and again["rechecked"] == 0
    assert {e["error"] for e in again["rejected"]} == {
        "already answered; to change a label, edit golden.csv after --finish"
    }
    b1, b2 = tools.golden_batch(session, 100), tools.golden_batch(session, 100)
    assert b1["items"] == b2["items"] and len(b1["items"]) == len(recheck_rows(session))
    item = b1["items"][0]
    sent = [{"id": item["id"], "answers": answer(item["text"])}]
    assert tools.golden_submit(session, sent)["accepted"] == 1
    retry = tools.golden_submit(session, sent)
    assert retry["accepted"] == 0 and "already answered" in retry["rejected"][0]["error"] and retry["rechecked"] == 1


def test_pass_one_ids_are_refused_while_second_answers_are_due(tmp_path):
    session, _ = start(tmp_path, 20)
    ids = first_pass(session)
    due = tools.golden_batch(session, 100)["items"]
    assert due and not {i["id"] for i in due} & set(ids.values())
    res = tools.golden_submit(session, [{"id": ids[i["text"]], "answers": answer(i["text"])} for i in due])
    assert res["accepted"] == 0 and res["rechecked"] == 0 and res["to_recheck"] == len(due)


def test_second_answer_ids_are_random_and_look_like_row_ids(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a, _ = start(tmp_path / "a", 20)
    b, _ = start(tmp_path / "b", 20)
    for session in (a, b):
        first_pass(session)
        tools.golden_batch(session, 100)
    ra, rb = recheck_rows(a), recheck_rows(b)
    assert [r["id"] for r in ra] == [r["id"] for r in rb]
    assert not {r["check_id"] for r in ra} & {r["check_id"] for r in rb}
    for r in ra + rb:
        assert r["check_id"] != r["id"] and r["check_id"][1:] not in golden_session.text_hash(r["text"])
        assert r["check_id"].startswith("g") and len(r["check_id"]) == len(r["id"])
        int(r["check_id"][1:], 16)
    taken = {"g" + "0" * 12}
    ids = iter(["0" * 12, "1" * 12])
    real = golden_session.secrets.token_hex
    golden_session.secrets.token_hex = lambda n: next(ids)
    try:
        assert golden_session._check_id(taken) == "g" + "1" * 12
    finally:
        golden_session.secrets.token_hex = real


def test_second_answers_are_mixed_in_with_texts_still_waiting(tmp_path):
    session, _ = start(tmp_path, 60)
    b = tools.golden_batch(session, 30)
    tools.golden_submit(session, [{"id": i["id"], "answers": answer(i["text"])} for i in b["items"]])
    mixed = tools.golden_batch(session, 100)
    rows = set(rows_ids(session))
    firsts = [i for i in mixed["items"] if i["id"] in rows]
    seconds = [i for i in mixed["items"] if i["id"] not in rows]
    assert len(firsts) == 30 and seconds and "pass" not in mixed
    assert mixed["remaining"] == len(firsts) + len(seconds)
    assert {i["text"] for i in seconds} <= {i["text"] for i in b["items"]}


def test_a_skip_needs_a_reason(tmp_path):
    session, _ = start(tmp_path, 6)
    ids = [i["id"] for i in tools.golden_batch(session)["items"]]
    res = tools.golden_submit(
        session, [{"id": ids[0], "skip": None}, {"id": ids[1], "skip": ""}, {"id": ids[2], "skip": 3}]
    )
    assert res["accepted"] == 0 and all("a skip needs a reason" in e["error"] for e in res["rejected"])


def test_written_examples_close_to_a_test_text_are_rejected(tmp_path):
    session, _ = start(tmp_path, 30)
    test = next(r["text"] for r in golden_session.load(session)["rows"] if r["split"] == "test")
    res = tools.golden_add(
        session,
        [
            {"text": test + " thanks", "answers": answer(test)},
            {
                "text": "completely different words about a laptop fan",
                "answers": {"team": "technical", "wants_refund": False},
            },
        ],
    )
    assert res["accepted"] == 1 and "shares most of its words with a held-out test text" in res["rejected"][0]["error"]


def test_finish_only_writes_next_to_the_session(tmp_path):
    session, _ = start(tmp_path, 6)
    first_pass(session)
    data = golden_session.load(session)
    assert data["out"] == "golden.csv"
    for bad in ("../escaped.csv", os.path.join("..", "x.csv"), "sub\\x.csv", "..", "", None):
        data["out"] = bad
        golden_session._save(session, data)
        with pytest.raises(ValueError, match="must be a file name next to the session file"):
            tools.golden_finish(session)
    assert not os.path.exists(tmp_path.parent / "escaped.csv")


# CLI


def run(capsys, *argv):
    code = cli.main(list(argv))
    return code, capsys.readouterr().out


def test_cli_golden_teacher_agent_and_finish(tmp_path, capsys):
    texts = tmp_path / "texts.txt"
    texts.write_text("\n".join(corpus(12)))
    out = str(tmp_path / "golden.csv")
    args = ["golden", str(texts), "--labels", ",".join(LABELS), "--teacher", "agent", "--strategy", "random"]
    code, text = run(capsys, *args, "-n", "8", "--out", out)
    assert code == cli.OK and "for a coding agent to label" in text and "--finish" in text
    code, text = run(capsys, *args, "--out", out, "--overwrite", "--json")
    payload = json.loads(text)
    assert code == cli.OK and payload["rows"] == 12 and payload["session"].endswith("golden.session.json")
    session = payload["session"]
    b = tools.golden_batch(session, 100)
    tools.golden_submit(session, [{"id": i["id"], "answers": {"label": truth(i["text"])["team"]}} for i in b["items"]])
    code, text = run(capsys, "golden", "--finish", session)
    assert code == cli.OK and "golden: wrote" in text and "12 rows" in text and "blind" not in text
    code, text = run(capsys, "golden", "--finish", session, "--out", str(tmp_path / "g2.csv"), "--json")
    assert code == cli.OK and json.loads(text)["rows"] == 12 and json.loads(text)["to_label"] == 0
    code, text = run(capsys, "golden", str(texts), "--finish", session, "--json")
    assert code == cli.INVALID and "only the session" in json.loads(text)["error"]["message"]
    code, text = run(capsys, "golden", str(texts), "--labels", "a,b", "--json")
    assert code == cli.INVALID and "--teacher agent" in json.loads(text)["error"]["message"]


# evaluate: accuracy on human labels, agreement with agent labels


def test_evaluate_reports_agreement_per_labeller(tmp_path):
    rows = [
        {
            "text": t,
            "label": truth(t)["team"],
            "split": "test",
            "labelled_by": "agent:claude-code" if i % 2 else "human",
        }
        for i, t in enumerate(corpus(12))
    ]
    rows[1]["label"] = next(x for x in LABELS if x != rows[1]["label"])
    model = ds.model(LABELS, FakeEngine(lambda t: {"label": truth(t)["team"]}, confidence=1.0))
    report = model.evaluate(rows)
    by = report.details["labelled_by"]
    assert by["human"] == {"decisions": 6, "accuracy": 1.0}
    assert by["agent:claude-code"] == {"decisions": 6, "agreement": 5 / 6}
    text = str(report)
    assert "by who labelled the rows" in text and "accuracy on rows labelled by human: 1.000" in text
    assert "agreement with agent:claude-code's labels: 0.833" in text
    plain = model.evaluate([{"text": r["text"], "label": r["label"]} for r in rows])
    assert "labelled_by" not in plain.details and "who labelled" not in str(plain)


# data_check


def test_data_check_finds_repeats_leaks_imbalance_and_bad_values(tmp_path):
    path = tmp_path / "d.csv"
    rows = [
        ("a", "you charged me twice", "billing", "test", "human"),
        ("b", "You charged me twice!", "billing", "train", "human"),
        ("c", "sync is broken", "technical", "", "agent:x"),
        ("d", "sync is broken", "sales", "", "agent:x"),
        ("e", "", "billing", "", ""),
        ("f", "hi", "legal", "", ""),
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "label", "split", "labelled_by"])
        w.writerows(rows)
    got = tools.data_check(str(path), labels=LABELS)
    assert got["rows"] == 6 and got["empty_texts"] == 1 and got["labelled"] == 4 and got["invalid"] == {"label": 1}
    assert got["balance"]["label"] == {"billing": 2, "technical": 1, "sales": 1}
    assert got["repeats"]["texts"] == 2 and got["repeats"]["labelled_differently"] == 1
    assert got["leaks"]["texts"] == 1 and got["leaks"]["examples"] == [["d.csv line 2", "d.csv line 3"]]
    assert got["lengths"]["min_words"] == 1 and got["lengths"]["under_3_words"] == 1
    assert got["labelled_by"] == {"human": 2, "agent:x": 2, "none": 1} and not got["ok"]
    advice = " ".join(got["advice"])
    for part in ("no text", "not options", "only 4 labelled rows", "fewer than 10", "more than once", "differently"):
        assert part in advice, part
    assert "in the test split and in training" in advice and got["near_copies"]["texts"] == 0
    loose = check(str(path))
    assert loose["balance"]["label"]["legal"] == 1 and not loose["invalid"]
    assert "no split=test" not in " ".join(loose["advice"])


def test_data_check_flags_near_copies_of_test_texts(tmp_path):
    path = tmp_path / "d.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "split"])
        w.writerow(["my card was charged twice for one order", "billing", "test"])
        w.writerow(["my card was charged twice for one order thanks", "billing", "train"])
        w.writerow(["the app crashes when I export a report", "technical", "train"])
    got = tools.data_check(str(path), labels=LABELS)
    assert got["near_copies"] == {"texts": 1, "examples": [["d.csv line 2", "d.csv line 3"]]} and not got["ok"]
    assert any("share at least 80% of their words with a test text" in a for a in got["advice"])


def test_data_check_clean_file_and_formats(tmp_path, toy_csv):
    got = tools.data_check(toy_csv, SCHEMA)
    assert got["labelled"] == 60 and "no split=test" in " ".join(got["advice"])
    jsonl = tmp_path / "d.jsonl"
    jsonl.write_text(
        "\n".join(
            json.dumps({"text": t, "answers": {"label": truth(t)["team"]}, "split": "test" if i < 30 else "train"})
            for i, t in enumerate(corpus(60))
        )
    )
    clean = tools.data_check(str(jsonl), labels=LABELS)
    assert clean["ok"] and clean["advice"][0].startswith("looks fine") and clean["split"] == {"test": 30, "train": 30}
    empty = tmp_path / "e.csv"
    empty.write_text("text,label\n")
    assert check(str(empty))["lengths"]["max_words"] == 0


# model tools


def test_evaluate_and_model_info_tools(tiny, tmp_path, monkeypatch):
    data = tmp_path / "test.csv"
    with open(data, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "split", "labelled_by"])
        w.writerows([t, truth(t)["team"], "test", "agent:claude-code"] for t in corpus(12))
    got = tools.evaluate(str(tiny), str(data), labels=LABELS, save=str(tmp_path / "models" / "team"))
    assert got["kind"] == "evaluate" and got["details"]["labelled_by"]["agent:claude-code"]["decisions"] == 12
    saved = got["saved"]
    assert saved.endswith("team-v1") and got["load"] == "ds.load(%r)" % saved
    info = tools.model_info(saved)
    assert info["labels"] == LABELS and info["fields"] == {"label": LABELS} and info["report"]["go"] is False
    again = tools.evaluate(saved, str(data))
    assert "saved" not in again and again["rows"]
    assert tools.evaluate(saved, str(data), labels=LABELS)["kind"] == "evaluate"
    with pytest.raises(ValueError, match="not a saved model folder"):
        tools.evaluate(str(tiny), str(data))
    os.remove(os.path.join(saved, "report.json"))
    assert tools.model_info(saved)["report"] is None
    calls = {}
    monkeypatch.setattr(
        "decisionsmith.training.finetuning.finetune",
        lambda data, model, **k: calls.setdefault("model", model) and ds.Report("finetune", "t", []),
    )
    tools.finetune(str(data), labels=LABELS)
    assert list(compile_schema(calls["model"]).fields) == ["label"]


def test_server_registers_the_build_tools():
    names = {fn.__name__ for fn in tools.TOOLS}
    assert {"golden_start", "golden_batch", "golden_submit", "golden_status", "golden_finish"} <= names
    assert {"golden_add", "data_check", "evaluate", "model_info"} <= names
    assert tools.server() is not None


# the whole flow, offline, driven by a scripted agent


def test_scripted_agent_builds_a_model_end_to_end(tiny, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with open("tickets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text"])
        w.writerows([t] for t in corpus(90))
    assert tools.data_check("tickets.csv")["rows"] == 90
    started = tools.golden_start("tickets.csv", labels=LABELS, n=90, agent="claude-code")
    session = started["session"]
    assert started["rows"] == 90 and started["test"] > 0 and session == "golden.session.json"
    first = golden_session.load(session)
    wrong = [r["text"] for r in first["rows"] if r["recheck"]][:1]
    while True:
        b = tools.golden_batch(session, 25)
        if not b["items"]:
            break
        second = second_ids(session)
        team = [truth(i["text"])["team"] for i in b["items"]]
        team = [
            next(x for x in LABELS if x != t) if i["id"] in second and i["text"] in wrong else t
            for i, t in zip(b["items"], team)
        ]
        tools.golden_submit(session, [{"id": i["id"], "answers": {"label": t}} for i, t in zip(b["items"], team)])
    st = tools.golden_status(session)
    assert st["disagreed"] == 1 and st["labelled"] == 90
    done = tools.golden_finish(session)
    assert done["rows"] == 90 and tools.data_check("golden.csv", session)["labelled"] == 89
    trained = tools.finetune("golden.csv", session, base=str(tiny), out="runs/v1")
    assert trained["kind"] == "finetune" and os.path.isdir("runs/v1")
    report = tools.evaluate("runs/v1", "golden.csv", session, save="models/team")
    assert report["details"]["labelled_by"]["agent:claude-code"]["decisions"] == report["details"]["rows"]
    model = ds.load(report["saved"])
    assert model.predict("you charged me twice") in LABELS
    assert tools.model_info(report["saved"])["report"]["go"] == report["go"]
