"""Every example runs offline: `DS_OFFLINE=1` answers LLM calls locally, `DS_LAYA` is the tiny checkpoint."""

import os
import runpy
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
KEYS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
    "TOGETHER_API_KEY",
    "FIREWORKS_API_KEY",
    "DEEPSEEK_API_KEY",
    "XAI_API_KEY",
    "MISTRAL_API_KEY",
    "TYPESAFE_API_KEY",
]


def scripts():
    only = os.environ.get("DS_EXAMPLES")
    for meta_path in sorted(EXAMPLES.rglob("meta.yaml")):
        if only and not meta_path.parent.relative_to(EXAMPLES).as_posix().startswith(only):
            continue
        meta = yaml.safe_load(meta_path.read_text()) or {}
        for py in sorted(meta_path.parent.glob("*.py")):
            if not py.name.startswith("_") and py.name != "schema.py":
                yield pytest.param(py, meta, id=str(py.relative_to(EXAMPLES)))


@pytest.mark.parametrize("path,meta", list(scripts()))
def test_example_runs(path, meta, tiny, tmp_path, monkeypatch, capsys):
    if meta.get("offline") is False:
        pytest.skip("needs %s" % meta.get("why", "resources this test can't provide"))
    for module in meta.get("imports", []):
        pytest.importorskip(module)
    monkeypatch.setenv("DS_OFFLINE", "1")
    monkeypatch.setenv("DS_LAYA", str(tiny))
    for key in KEYS + list(meta.get("keys", [])):
        monkeypatch.setenv(key, os.environ.get(key) or "offline")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(path.parent))
    before = set(sys.modules)
    try:
        runpy.run_path(str(path), run_name="__main__")
    finally:
        for name in set(sys.modules) - before:
            if str(path.parent) in str(getattr(sys.modules[name], "__file__", "") or ""):
                del sys.modules[name]
    if meta.get("expect"):
        assert meta["expect"] in capsys.readouterr().out


def test_every_example_has_meta_and_the_gallery_is_current():
    for folder in {p.parent for p in EXAMPLES.rglob("*.py")} - {EXAMPLES}:
        assert (folder / "meta.yaml").exists(), folder
    sys.path.insert(0, str(ROOT / "scripts"))
    import gen_gallery

    assert gen_gallery.main(["--check"]) == 0
