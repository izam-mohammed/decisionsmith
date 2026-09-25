import asyncio
import types

import pytest

import decisionsmith as ds
from decisionsmith.engines import EngineError
from decisionsmith.integrations import _base
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, truth


class AsyncFake(FakeEngine):
    """A FakeEngine with a native `aask`, counting calls."""

    def __init__(self, *a, broken=None, **k):
        super().__init__(*a, **k)
        self.async_calls, self.broken = 0, broken

    async def aask(self, text, questions):
        self.async_calls += 1
        if self.broken == "raise":
            raise EngineError(self.name, "down")
        if self.broken == "none":
            return None
        return self.ask(text, questions)


def test_adecide_matches_decide(db):
    teacher = AsyncFake(truth, confidence=1.0, name="teacher")
    student = FakeEngine(truth, confidence=0.95, name="student")
    h = ds.harness(Ticket, teacher=teacher, student=student, mode="shadow", log=db)
    r = asyncio.run(h.adecide("you charged me twice, refund please"))
    assert r.value == h.decide("you charged me twice, refund please").value
    assert r.source == {"team": "teacher", "wants_refund": "teacher"} and teacher.async_calls == 1
    assert asyncio.run(h.acall("the app crashes on login")).team == "technical"
    assert len(h.log.rows("Ticket")) == 3


def test_adecide_cascade_asks_the_teacher_when_unsure(db):
    teacher = AsyncFake(truth, confidence=1.0, name="teacher")
    student = AsyncFake(truth, confidence=0.5, name="student")
    h = ds.harness(Ticket, teacher=teacher, student=student, log=db, audit=0)
    r = asyncio.run(h.adecide("how much is the plan"))
    assert r.source == {"team": "teacher", "wants_refund": "teacher"} and student.async_calls == 1
    assert teacher.async_calls == 1


def test_adecide_falls_back_when_an_engine_fails(db):
    for broken in ("raise", "none"):
        student = AsyncFake(truth, confidence=0.99, name="student", broken=broken)
        h = ds.harness(Ticket, teacher=FakeEngine(truth, name="teacher"), student=student, log=None)
        r = asyncio.run(h.adecide("how much is the plan"))
        assert set(r.source.values()) == {"teacher"}
    h = ds.harness(Ticket, student=AsyncFake(name="s", broken="raise"), log=None)
    with pytest.raises(EngineError, match="no engine could answer"):
        asyncio.run(h.adecide("x"))
    with pytest.raises(ValueError, match="non-empty"):
        asyncio.run(h.adecide(" "))


def test_apredict():
    labels = ["billing", "technical", "sales"]
    fake = AsyncFake(lambda t: {"label": truth(t)["team"]}, name="f")
    m = ds.model(labels, fake)
    assert asyncio.run(m.apredict("you charged me twice")) == "billing"
    assert asyncio.run(m.apredict(["sync is broken", "can I get a quote"])) == ["technical", "sales"]
    assert fake.async_calls == 3
    plain = ds.model(Ticket, FakeEngine(truth))
    assert asyncio.run(plain.apredict("refund my card charge")).wants_refund is True
    with pytest.raises(ValueError, match="non-empty"):
        asyncio.run(plain.apredict([""]))


def test_base_resolve(db):
    m = ds.model(["a", "b"], FakeEngine(name="f"))
    d = _base.resolve(m)
    assert d.simple and d.fields == ["label"] and d.name == "f" and d.field(d.call("x")) in ("a", "b")
    assert asyncio.run(d.acall("x")) in ("a", "b")
    h = ds.harness(Ticket, teacher=FakeEngine(truth, name="t"), student=FakeEngine(truth, name="s"), log=None)
    d = _base.resolve(h)
    assert not d.simple and d.name == "s+t" and d.fields == ["team", "wants_refund"]
    assert d.field(d.call("my invoice is wrong"), "team") == "billing"
    with pytest.raises(TypeError, match="expected ds.model"):
        _base.resolve("nope")


def test_run_sync_and_names():
    async def value():
        return 5

    async def fail():
        raise KeyError("x")

    assert _base.run_sync(value) == 5

    async def inside():
        return _base.run_sync(value)

    assert asyncio.run(inside()) == 5

    async def inside_fail():
        return _base.run_sync(fail)

    with pytest.raises(KeyError):
        asyncio.run(inside_fail())
    assert _base.model_name(types.SimpleNamespace(model_name="gpt", model="x")) == "gpt"
    assert _base.model_name(types.SimpleNamespace(model=None, model_id="m")) == "m"
    assert _base.model_name(object()) == "object"
    e = _base.teacher("t", lambda s, u: '{"q0": "billing", "q1": true}')
    assert e.name == "t" and e.ask("x", ds.harness(Ticket, teacher="fake", log=None).schema.questions())
