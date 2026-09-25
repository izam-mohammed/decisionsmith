import enum
from typing import Annotated, Literal, Optional

import pytest
from pydantic import BaseModel, Field

import decisionsmith as ds
from decisionsmith.schema import MAX_OPTIONS, compile_schema, confidence, top
from tests.conftest import Ticket


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


class Everything(BaseModel):
    color: Color
    size: Literal[1, 2, 3]
    ok: Literal[True, False]
    urgency: Annotated[Literal["low", "medium", "high"], ds.Scale, ds.Options(high="drop everything")]
    note: str = ""
    maybe: Optional[bool] = None  # noqa: UP045


def test_ticket_questions():
    s = compile_schema(Ticket)
    q = s.questions()
    assert q["team"] == {
        "type": "choice",
        "instructions": "A support ticket. Which team should handle this?",
        "criteria": {"billing": "payments and refunds", "technical": "", "sales": "pricing"},
    }
    assert q["wants_refund"] == {
        "type": "noul",
        "instructions": "A support ticket. Does the customer ask for a refund?",
    }
    assert s.name == "Ticket" and len(s.fingerprint) == 12
    assert list(s.questions(["team"])) == ["team"]


def test_every_type_and_skipped_fields():
    s = compile_schema(Everything)
    assert list(s.fields) == ["color", "size", "ok", "urgency"]
    assert s.fields["color"].labels == ("red", "blue")
    assert s.fields["size"].labels == ("1", "2", "3")
    assert s.fields["ok"].kind == "bool"
    u = s.fields["urgency"]
    assert u.kind == "scale" and u.question["criteria"] == ["low", "medium", "high: drop everything"]
    assert s.fields["color"].question["criteria"] == ["red", "blue"]
    assert s.fields["urgency"].question["instructions"] == "Rate the urgency."
    assert s.fields["ok"].question["instructions"] == "Is this true: ok?"
    assert s.fields["color"].question["instructions"] == "What is the color?"


def test_bool_descriptions():
    class M(BaseModel):
        spam: Annotated[bool, ds.Options(true="it is spam", false="it is fine")]

    assert compile_schema(M).questions()["spam"]["criteria"] == {"true": "it is spam", "false": "it is fine"}


def test_distribution_and_build():
    s = compile_schema(Everything)
    d = {
        "color": s.distribution("color", {"probabilities": {"red": 0.2, "blue": 0.6}}),
        "size": s.distribution("size", {"choice": "2"}),
        "ok": s.distribution("ok", {"noul": 0.7}),
        "urgency": s.distribution("urgency", {"probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}),
    }
    assert d["color"] == pytest.approx({"red": 0.25, "blue": 0.75})
    assert d["size"] == {"1": 0.0, "2": 1.0, "3": 0.0}
    assert d["ok"] == pytest.approx({"false": 0.3, "true": 0.7})
    m = s.build(d)
    assert m.color is Color.BLUE and m.size == 2 and m.ok is True and m.urgency == "high" and m.note == ""
    assert confidence(d["ok"]) == pytest.approx(0.7) and top(d["color"]) == "blue" and confidence({}) == 0.0
    assert s.distribution("ok", {"probabilities": {"true": 0.4}}) == pytest.approx({"false": 0.6, "true": 0.4})
    assert s.distribution("urgency", {"score": 1.4})["medium"] == 1.0
    assert s.distribution("urgency", {"score": 9})["high"] == 1.0
    assert s.distribution("color", {"probabilities": {"red": 0, "blue": 0}, "choice": "red"})["red"] == 1.0


@pytest.mark.parametrize(
    "name,answer",
    [
        ("color", "nope"),
        ("color", {"choice": "green"}),
        ("urgency", {"score": "x"}),
        ("urgency", {"score": float("nan")}),
        ("ok", {"noul": "yes"}),
        ("ok", {}),
        ("color", {"probabilities": {"red": True}}),
    ],
)
def test_bad_answers(name, answer):
    with pytest.raises(ValueError):
        compile_schema(Everything).distribution(name, answer)


def test_label_of():
    s = compile_schema(Everything)
    assert s.label_of("ok", True) == "true"
    assert s.label_of("ok", "No") == "false"
    assert s.label_of("ok", "yes") == "true"
    assert s.label_of("ok", "1") == "true" and s.label_of("ok", "0") == "false"
    assert s.label_of("ok", "FALSE") == "false"
    assert s.label_of("color", Color.RED) == "red"
    assert s.label_of("color", "BLUE") == "blue"
    assert s.label_of("color", "blue") == "blue"
    assert s.label_of("size", 3) == "3"
    with pytest.raises(ValueError):
        s.label_of("ok", "perhaps")
    with pytest.raises(ValueError):
        s.label_of("color", "green")
    with pytest.raises(KeyError):
        s.label_of("nope", 1)


def _bad(annotation, **kw):
    return type("Bad", (BaseModel,), {"__annotations__": {"x": annotation}, **kw})


@pytest.mark.parametrize(
    "model,msg",
    [
        (_bad(str), "can't be decided"),
        (_bad(Optional[bool]), "Optional"),  # noqa: UP045
        (_bad(Literal["a"]), "at least two"),
        (_bad(Annotated[Literal["a", "b"], ds.Options(c="?")]), "unknown options"),
        (_bad(Literal[tuple("o%d" % i for i in range(MAX_OPTIONS + 1))]), "more than"),
        (_bad(str, x=""), "no decision fields"),
        (_bad(Literal[1, "1"]), "unique"),
    ],
)
def test_bad_schemas(model, msg):
    with pytest.raises(TypeError, match=msg):
        compile_schema(model)


def test_not_a_model():
    with pytest.raises(TypeError, match="Pydantic model"):
        compile_schema(dict)  # type: ignore[arg-type]


def test_many_options_warns():
    with pytest.warns(UserWarning, match="lose accuracy"):
        compile_schema(_bad(Literal[tuple("w%d" % i for i in range(21))]))


@pytest.mark.parametrize("levels", [("one",), ("a", "a"), ("a", 1)])
class Level(enum.Enum):
    LOW = "low"
    HIGH = "high"


def test_scale_forms():
    class M(BaseModel):
        a: Annotated[Level, ds.Scale]
        b: Annotated[Literal["x", "y"], ds.Scale()]

    s = compile_schema(M)
    assert s.fields["a"].kind == "scale" and s.fields["b"].kind == "scale"
    assert s.build({"a": {"low": 0.2, "high": 0.8}, "b": {"x": 1.0, "y": 0.0}}).a is Level.HIGH
    with pytest.raises(TypeError, match="not bool"):
        compile_schema(_bad(Annotated[bool, ds.Scale]))


def test_options_repr_and_no_docstring():
    assert repr(ds.Options(a="x")) == "Options(a='x')"

    class Plain(BaseModel):
        x: bool = Field(description="Is it?")

    assert compile_schema(Plain).questions()["x"]["instructions"] == "Is it?"
