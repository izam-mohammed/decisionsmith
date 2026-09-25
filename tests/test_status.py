import decisionsmith as ds
from decisionsmith.schema import compile_schema
from decisionsmith.status import CASCADE_ROWS, FINETUNE_ROWS, STUDENT_ROWS, compute_status
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, corpus, truth

S = compile_schema(Ticket)
LABELS = S.fields["team"].labels


def row(i, agree=True, conf=0.95, teacher=True, human=None, student="s", team=None):
    team = team or LABELS[i % 3]
    other = LABELS[(LABELS.index(team) + 1) % 3]
    pick = team if agree else other
    sd = {k: (conf if k == pick else (1 - conf) / 2) for k in LABELS}
    td = {k: float(k == team) for k in LABELS}
    return {
        "id": "r%d" % i,
        "student": student,
        "student_dists": {"team": sd},
        "teacher_dists": {"team": td} if teacher else None,
        "labels": {"team": human} if human else {},
    }


def status(rows, student="s", mode="shadow", threshold=0.8):
    return compute_status(S, rows, {n: mode for n in S.fields}, student, lambda n: threshold).fields["team"]


def test_ready_for_student_and_cascade():
    f = status([row(i) for i in range(STUDENT_ROWS)])
    assert f.advice == "ready for student" and f.agreement == 1.0 and f.accuracy_source == "teacher"
    assert status([row(i) for i in range(STUDENT_ROWS)], mode="student").advice.startswith("in student mode")
    rows = [row(i, agree=i % 20 != 0) for i in range(CASCADE_ROWS)]
    f = status(rows)
    assert f.advice == "ready for cascade" and 0.9 <= f.agreement < 0.97
    assert status(rows, mode="cascade").advice == "cascade is right; keep auditing"


def test_human_labels_win_when_enough():
    rows = [row(i, human=LABELS[(i + 1) % 3]) for i in range(40)]
    f = status(rows)
    assert f.accuracy_source == "human" and f.accuracy_when_sure == 0.0 and f.human_labels == 40


def test_finetune_and_disagreement_advice():
    rows = [row(i, agree=i % 2 == 0) for i in range(FINETUNE_ROWS)]
    assert status(rows).advice == "ready to finetune: run h.finetune()"
    rows = [row(i, agree=i % 2 == 0, team="billing") for i in range(CASCADE_ROWS)]
    assert status(rows).advice.startswith("student disagrees often")
    assert status(rows[:10]).advice == "needs more data (have 10, want %d)" % CASCADE_ROWS


def test_no_student_and_teacher_mode():
    assert status([row(0)], student=None).advice.startswith("no student yet")
    f = status([row(i, teacher=False) for i in range(5)], mode="teacher")
    assert f.advice == "use mode='shadow' to measure the student" and f.accuracy_when_sure is None
    assert status([row(0, student="other")]).rows == 0


def test_status_from_harness_prints(db):
    h = ds.harness(
        Ticket, teacher=FakeEngine(truth, name="t"), student=FakeEngine(truth, name="s"), mode="shadow", log=db
    )
    h.many(corpus(6))
    st = h.status()
    text = str(st)
    assert "team:" in text and "student agrees 100%" in text and "6 labelled" in text and repr(st) == text
    assert st.to_dict()["fields"]["team"]["both"] == 6
    bare = compute_status(S, [], {n: "teacher" for n in S.fields}, None, lambda n: 0.8)
    assert "0 labelled" in str(bare)
