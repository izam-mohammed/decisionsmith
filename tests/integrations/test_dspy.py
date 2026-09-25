import asyncio

import pytest

dspy = pytest.importorskip("dspy")

import decisionsmith as ds  # noqa: E402
from decisionsmith.engines import from_string  # noqa: E402
from decisionsmith.integrations.dspy import _text  # noqa: E402
from decisionsmith.schema import compile_schema  # noqa: E402
from tests.conftest import Ticket  # noqa: E402
from tests.integrations.openai_mock import GOOD  # noqa: E402

Q = compile_schema(Ticket).questions()


def test_dspy_lm_is_a_teacher():
    lm = dspy.LM("openai/gpt-4o-mini", api_key="sk-test", cache=False, mock_response=GOOD)
    e = from_string(lm)
    assert e.name == "dspy:openai/gpt-4o-mini"
    assert e.ask("you charged me twice", Q)["answers"]["team"]["choice"] == "billing"
    assert asyncio.run(e.aask("x", Q))["answers"]["wants_refund"]["noul"] == 1.0
    m = ds.model(Ticket, "fake")
    assert m.label(["my invoice is wrong"], lm)[0]["answers"]["team"] == {
        "billing": 1.0,
        "technical": 0.0,
        "sales": 0.0,
    }


def test_reply_shapes():
    assert _text([{"text": "a"}]) == "a" and _text(["b"]) == "b" and _text([]) == ""


def harness():
    from decisionsmith.testing import FakeEngine
    from tests.conftest import truth

    return ds.harness(Ticket, teacher=FakeEngine(truth, name="t"), log=None)


def test_decision_module_returns_a_prediction():
    from decisionsmith.integrations.dspy import DecisionModule
    from decisionsmith.testing import FakeEngine

    program = DecisionModule(harness())
    assert isinstance(program, dspy.Module) and DecisionModule is type(program)
    p = program(text="refund my card charge")
    assert isinstance(p, dspy.Prediction) and p.team == "billing" and p.wants_refund is True
    assert asyncio.run(program.acall(text="sync is broken")).team == "technical"
    labels = DecisionModule(ds.model(["spam", "ham"], FakeEngine(lambda t: {"label": "spam"})), input="message")
    assert labels(message="win money").label == "spam"
    with pytest.raises(AttributeError):
        import decisionsmith.integrations.dspy as m

        m.Nope


def test_metric_in_dspy_evaluate():
    from decisionsmith.integrations.dspy import DecisionModule, metric

    devset = [
        dspy.Example(text="my invoice is wrong", team="billing", wants_refund=False).with_inputs("text"),
        dspy.Example(text="sync is broken", team="billing", wants_refund=False).with_inputs("text"),
    ]
    program = DecisionModule(harness())
    assert dspy.Evaluate(devset=devset, metric=metric("team"), num_threads=1)(program).score == 50.0
    both = dspy.Evaluate(devset=devset, metric=metric(), num_threads=1)(program)
    assert [r[2] for r in both.results] == [1.0, 0.5]
    m = metric()
    assert m(devset[0], program(text="my invoice is wrong"), trace=[]) is True
    assert m(devset[1], program(text="sync is broken"), trace=[]) is False
    assert m(dspy.Example(text="x").with_inputs("text"), dspy.Prediction()) == 0.0
    assert metric("wants_refund")(dspy.Example(wants_refund="True"), dspy.Prediction(wants_refund=True)) == 1.0
