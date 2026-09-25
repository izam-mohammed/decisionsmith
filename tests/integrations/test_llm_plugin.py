import asyncio

import pytest

llm = pytest.importorskip("llm")

from click.testing import CliRunner  # noqa: E402

import decisionsmith as ds  # noqa: E402
from decisionsmith.engines import from_string  # noqa: E402
from decisionsmith.integrations import llm_plugin  # noqa: E402
from decisionsmith.schema import compile_schema  # noqa: E402
from decisionsmith.testing import FakeEngine  # noqa: E402
from tests.conftest import Ticket  # noqa: E402
from tests.integrations.openai_mock import GOOD  # noqa: E402

Q = compile_schema(Ticket).questions()


class Canned(llm.Model):
    model_id = "canned"

    def __init__(self, reply=GOOD):
        self.reply = reply

    def execute(self, prompt, stream, response, conversation):
        assert prompt.system and "<text>" in prompt.prompt
        response.set_usage(input=11, output=3)
        yield self.reply


class AsyncCanned(llm.AsyncModel):
    model_id = "canned-async"

    async def execute(self, prompt, stream, response, conversation):
        response.set_usage(input=5, output=2)
        yield GOOD


@pytest.fixture(autouse=True)
def user_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))


def test_llm_models_are_teachers():
    e = from_string(Canned())
    assert e.name == "llm:canned"
    out = e.ask("you charged me twice", Q)
    assert out["answers"]["team"]["choice"] == "billing" and out["usage"] == {"input_tokens": 11, "output_tokens": 3}
    a = llm_plugin.teacher(AsyncCanned())
    assert asyncio.run(a.aask("x", Q))["usage"] == {"input_tokens": 5, "output_tokens": 2}
    assert a.ask("x", Q)["answers"]["wants_refund"]["noul"] == 1.0


def test_the_plugin_hook_is_lazy_and_registers_decide():
    assert "register_commands" in dir(llm_plugin)
    with pytest.raises(AttributeError):
        llm_plugin.nothing
    from llm.cli import cli
    from llm.plugins import pm

    assert "decide" in cli.commands, "the llm entry point loads decisionsmith.integrations.llm_plugin"
    assert any(getattr(p, "__name__", "") == "decisionsmith.integrations.llm_plugin" for p in pm.get_plugins())


def test_decide_command(monkeypatch):
    from llm.cli import cli

    student = FakeEngine(lambda t: {"label": "billing" if "invoice" in t else "sales"}, confidence=0.6)
    real = ds.model
    monkeypatch.setattr(ds, "model", lambda labels, engine, question=None: real(labels, student, question=question))
    run = CliRunner().invoke
    out = run(cli, ["decide", "my invoice is wrong", "-l", "billing", "-l", "sales", "--question", "Which team?"])
    assert out.exit_code == 0, out.output
    assert out.output.strip() == "billing"
    assert run(cli, ["decide", "-l", "billing", "-l", "sales"], input="hello\n").output.strip() == "sales"
    monkeypatch.setattr(llm, "get_model", lambda name: Canned('{"q0": "sales"}'))
    out = run(cli, ["decide", "my invoice is wrong", "-l", "billing", "-l", "sales", "--teacher", "canned"])
    assert out.exit_code == 0, out.output
    assert out.output.strip() == "sales"
