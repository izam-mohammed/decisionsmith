import json
import math
import threading

import numpy as np
import pytest

from decisionsmith.engines import EngineError
from decisionsmith.log import Log
from decisionsmith.report import Report, metrics
from decisionsmith.testing import FakeEngine
from decisionsmith.training import calibrate


def test_fit_temperature_recovers_known_value():
    rng = np.random.default_rng(0)
    logits, targets = [], []
    for _ in range(2000):
        z = rng.normal(size=4) * 3
        p = np.exp(z / 2.0) / np.exp(z / 2.0).sum()
        logits.append(z)
        targets.append(np.eye(4)[rng.choice(4, p=p)])
    assert calibrate.fit_temperature(logits, targets) == pytest.approx(2.0, abs=0.15)
    assert calibrate.fit_temperature([], []) == 1.0
    assert calibrate.fit_temperature([[10.0, 0.0]], [[0.0, 1.0]]) == pytest.approx(calibrate.TEMP_MAX, abs=1e-3)
    assert calibrate.fit_temperature([[1.0, 0.0]], [[1.0, 0.0]]) == pytest.approx(calibrate.TEMP_MIN, abs=1e-3)


def test_scale_ece_threshold():
    assert calibrate.scale([0.5, 0.5], 2.0) == pytest.approx([0.5, 0.5])
    assert calibrate.scale([0.8, 0.2], 1.0) == pytest.approx([0.8, 0.2])
    assert calibrate.scale([0.8, 0.2], 2.0)[0] < 0.8
    assert calibrate.ece([0.9] * 10, [1] * 9 + [0]) == pytest.approx(0.0)
    assert calibrate.ece([1.0] * 4, [0] * 4) == pytest.approx(1.0)
    assert math.isnan(calibrate.ece([], []))
    assert calibrate.ece([0.0, 0.5], [0, 1]) == pytest.approx(0.25)
    conf = [0.99] * 30 + [0.7] * 30 + [0.6] * 30
    right = [1] * 30 + [1] * 28 + [0] * 2 + [0] * 30
    assert calibrate.threshold(conf, right, 0.95) == pytest.approx(0.7)
    assert calibrate.threshold(conf, right, 0.999) == pytest.approx(0.99)
    assert calibrate.threshold(conf, right, 1.0, min_rows=31) is None
    assert calibrate.threshold([], [], 0.9) is None


def test_metrics_and_report(tmp_path):
    m = metrics([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]], [[1, 0], [0, 1], [0, 1]])
    assert m["n"] == 3 and m["accuracy"] == pytest.approx(2 / 3)
    assert m["macro_f1"] == pytest.approx(2 / 3)
    assert m["curve"][0] == {"threshold": 0.5, "coverage": 1.0, "accuracy": pytest.approx(2 / 3)}
    assert m["curve"][-1]["accuracy"] is None
    assert metrics([], []) == {"n": 0}
    o = metrics([[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]], ordinal=True)
    assert o["mae"] == pytest.approx(1.0) and o["macro_f1"] == 0.0
    r = Report(
        "bench",
        "t",
        [{"a": 1.23456, "b": None, "c": True, "d": float("nan"), "e": 123.0}, {"f": "x"}],
        go=False,
        reasons=["why"],
        path="p",
    )
    text = str(r)
    assert "1.235" in text and "yes" in text and "123.0" in text and "go: no" in text and "  - why" in text
    assert "saved: p" in text and repr(r) == text
    assert r.columns == ["a", "b", "c", "d", "e", "f"]
    assert json.loads(open(r.save(str(tmp_path / "r.json"))).read())["go"] is False
    assert "<table>" in open(r.save(str(tmp_path / "r.html"))).read()
    assert "go" not in str(Report("adapt", "t", []))
    assert "go: yes" in Report("x", "t", [], go=True).html()


def test_log(tmp_path):
    log = Log(tmp_path / "sub" / "x.db")
    log.add(
        {
            "id": "a",
            "schema": "S",
            "text": "hi",
            "value": {"x": "1"},
            "source": {"x": "teacher"},
            "teacher_dists": {"x": {"1": 1.0}},
            "student_dists": None,
            "latency_ms": 1.0,
        }
    )
    log.add({"id": "b", "schema": "Other", "text": "yo"})
    log.label("a", {"x": "1"})
    with pytest.raises(KeyError):
        log.label("zzz", {"x": "1"})
    rows = log.rows("S")
    assert len(rows) == 1 and rows[0]["labels"] == {"x": "1"} and rows[0]["student_dists"] is None
    assert rows[0]["teacher_dists"] == {"x": {"1": 1.0}}
    log.mark_trained(["a", "a"], "run1")
    assert log.trained() == {"a"}
    assert log.get("k", 5) == 5
    log.set("k", {"v": 1})
    assert log.get("k") == {"v": 1}
    threads = [
        threading.Thread(target=log.add, args=({"id": "t%d" % i, "schema": "S", "text": "x"},)) for i in range(20)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(log.rows("S")) == 21
    log.close()


Q = {
    "team": {"type": "choice", "instructions": "?", "criteria": {"billing": "", "sales": ""}},
    "ok": {"type": "noul", "instructions": "?"},
    "lvl": {"type": "score", "instructions": "?", "criteria": ["low", "high: very"]},
    "one": {"type": "choice", "instructions": "?", "criteria": ["only"]},
}


def test_fake_engine():
    fake = FakeEngine({"a": {"team": "sales", "ok": True, "lvl": "high", "one": "only"}}, confidence=0.8)
    out = fake.ask("a", Q)
    a = out["answers"]
    assert a["team"]["choice"] == "sales" and a["team"]["probabilities"]["sales"] == pytest.approx(0.8)
    assert a["ok"]["noul"] == pytest.approx(0.8)
    assert a["lvl"]["probabilities"]["1"] == pytest.approx(0.8) and a["lvl"]["score"] == pytest.approx(0.8)
    assert a["one"]["probabilities"] == {"only": 1.0}
    assert out["usage"]["input_tokens"] == 1 and fake.calls == [("a", list(Q))]
    assert fake.ask("unknown", Q)["answers"]["team"]["choice"] in ("billing", "sales")
    assert FakeEngine({"a": {"lvl": 0}}).ask("a", Q)["answers"]["lvl"]["probabilities"]["0"] == pytest.approx(0.9)
    wrong = FakeEngine(lambda t: {"team": "sales"}, accuracy=0.0, confidence=lambda t, q: 0.6)
    assert wrong.ask("x", Q)["answers"]["team"] == {
        "type": "choice",
        "probabilities": {"billing": 0.6, "sales": pytest.approx(0.4)},
        "confidence": 0.6,
        "choice": "billing",
    }
    with pytest.raises(ValueError, match="not a level"):
        FakeEngine({"a": {"lvl": "mid"}}).ask("a", Q)
    with pytest.raises(EngineError, match="down"):
        FakeEngine(error="down").ask("a", Q)
    with pytest.raises(TimeoutError):
        FakeEngine(error=TimeoutError()).ask("a", Q)
    FakeEngine(latency=0.001).ask("a", Q)
    many = [
        FakeEngine(lambda t: {"team": "sales"}, accuracy=0.5).ask(str(i), Q)["answers"]["team"]["choice"]
        for i in range(200)
    ]
    assert 60 < many.count("sales") < 140


def test_public_functions_survive_submodule_imports():
    import importlib
    import pkgutil
    import types

    import decisionsmith as ds

    names = [m.name for m in pkgutil.walk_packages(ds.__path__, "decisionsmith.")]
    assert {"decisionsmith.core", "decisionsmith.engines.base", "decisionsmith.training.finetuning"} <= set(names)
    for mod in names:
        importlib.import_module(mod)
    assert isinstance(ds.finetune, types.FunctionType) and isinstance(ds.bench, types.FunctionType)
    assert isinstance(ds.harness, types.FunctionType) and isinstance(ds.model, types.FunctionType)
    assert isinstance(ds.testing, types.ModuleType) and ds.testing.__name__ == "decisionsmith.testing"
    assert sorted(ds.__all__) == sorted(
        [
            "Engine",
            "EngineError",
            "LLM",
            "Model",
            "Options",
            "Report",
            "Result",
            "Scale",
            "Status",
            "__version__",
            "bench",
            "finetune",
            "harness",
            "model",
            "testing",
        ]
    )


LITE = r"""
import builtins, csv, sys

HEAVY = {
    "numpy", "torch", "laya", "transformers", "litellm", "instructor", "mcp", "httpx", "httpx2", "anthropic",
    "openai", "langchain", "langchain_core", "langgraph", "llama_index", "dspy", "crewai", "pydantic_ai", "haystack",
    "smolagents", "autogen_core", "autogen_ext", "autogen_agentchat", "semantic_kernel", "agents", "claude_agent_sdk",
    "google", "agno", "guardrails", "nemoguardrails", "outlines", "marvin", "llm", "chromadb", "qdrant_client",
    "pandas", "polars", "datasets", "duckdb", "pyspark", "dask", "ray", "label_studio_sdk", "argilla", "cleanlab",
    "huggingface_hub", "opentelemetry", "langfuse", "langsmith", "phoenix", "prometheus_client", "mlflow", "wandb",
    "deepeval", "inspect_ai", "fastapi", "starlette", "django", "flask", "litestar", "celery", "rq", "dramatiq",
    "gradio", "streamlit", "instructor", "portkey_ai", "click", "pluggy", "ollama", "outlines_core",
}
real = builtins.__import__


def guard(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".")[0] in HEAVY:
        raise ImportError("blocked " + name)
    return real(name, globals, locals, fromlist, level)


builtins.__import__ = guard
from typing import Literal

from pydantic import BaseModel

import decisionsmith as ds
from decisionsmith.testing import FakeEngine


class T(BaseModel):
    team: Literal["billing", "sales"]
    urgent: bool


def truth(text):
    return {"team": "billing" if "bill" in text else "sales", "urgent": "now" in text}


texts = ["bill %d now" % i if i % 2 else "price %d" % i for i in range(260)]
h = ds.harness(T, teacher=FakeEngine(truth), student=FakeEngine(truth, accuracy=0.9, seed=1), mode="shadow",
               log=sys.argv[1] + "/d.db")
assert h(texts[1]).team == "billing"
h.many(texts[2:])
r = h.decide(texts[0])
h.label(r.id, team="sales")
assert "student agrees" in str(h.status())
assert h.adapt().rows[0]["temperature"] > 0
path = sys.argv[1] + "/b.csv"
with open(path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["text", "team", "urgent"])
    w.writerows([t, truth(t)["team"], truth(t)["urgent"]] for t in texts[:30])
assert ds.bench(T, path, [FakeEngine(truth)]).rows[0]["accuracy"] == 1.0
assert ds.model(["a", "b"], FakeEngine()).predict("x") in ("a", "b")
import importlib, pkgutil
import decisionsmith.integrations as integrations
for info in pkgutil.iter_modules(integrations.__path__):
    importlib.import_module("decisionsmith.integrations." + info.name)
loaded = sorted(m for m in sys.modules if m.split(".")[0] in HEAVY)
assert not loaded, loaded
print("ok")
"""


def test_core_paths_need_no_heavy_dependencies(tmp_path):
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-c", LITE, str(tmp_path)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr
