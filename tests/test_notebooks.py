"""Every notebook runs top to bottom offline (`DS_OFFLINE=1`, tiny Laya checkpoint), so none of them rot.

Code cells run in one namespace; `%%writefile` cells write their file; other `!`/`%` lines are skipped.
Only the notebooks in GPU_ONLY (metadata `"decisionsmith": {"offline": false}`) are skipped; every other one must run.
"""

import json
import os
from pathlib import Path

import pytest

from tests.test_examples import KEYS

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
GPU_ONLY = {"04_finetune_full_gpu.ipynb", "06_typed_decisions_reproduction.ipynb"}


def cells(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return nb.get("metadata", {}).get("decisionsmith", {}), [c for c in nb["cells"] if c["cell_type"] == "code"]


def run_cell(source, namespace):
    lines = source.splitlines()
    if lines and lines[0].startswith("%%writefile"):
        Path(lines[0].split(maxsplit=1)[1].strip()).write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
        return
    if lines and lines[0].startswith("%%"):
        return
    code = "\n".join(line for line in lines if not line.lstrip().startswith(("!", "%")))
    exec(compile(code, "<cell>", "exec"), namespace)


@pytest.mark.parametrize("path", sorted(NOTEBOOKS.glob("*.ipynb")), ids=lambda p: p.name)
def test_notebook_runs_offline(path, tiny, tmp_path, monkeypatch, no_network):
    meta, code = cells(path)
    assert (meta.get("offline") is False) == (path.name in GPU_ONLY)
    if path.name in GPU_ONLY:
        pytest.skip(meta["why"])
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("DS_OFFLINE", "1")
    monkeypatch.setenv("DS_LAYA", str(tiny))
    for key in KEYS:
        monkeypatch.setenv(key, os.environ.get(key) or "offline")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    namespace = {"__name__": "__main__"}
    for cell in code:
        run_cell("".join(cell["source"]), namespace)


def test_cell_runner(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns = {}
    run_cell("%%writefile a.py\nX = 1", ns)
    run_cell("%%time\nboom()", ns)
    run_cell("!pip install x\n%matplotlib inline\ny = 2", ns)
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "X = 1\n" and ns["y"] == 2


@pytest.mark.parametrize("path", sorted(NOTEBOOKS.glob("*.ipynb")), ids=lambda p: p.name)
def test_notebook_header(path):
    text = path.read_text(encoding="utf-8")
    nb = json.loads(text)
    intro = "".join(nb["cells"][0]["source"])
    assert "Runs on:" in intro and "Nandakishor M / Convai Innovations" in intro
    installs = [line for c in nb["cells"] for line in "".join(c["source"]).splitlines() if "pip install" in line]
    assert installs and all("uv pip install" in line for line in installs)
    assert not any(p in text for p in ("/Users/", "/home/", "/private/", "C:\\\\Users"))
