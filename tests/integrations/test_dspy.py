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
