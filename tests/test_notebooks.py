"""Every notebook runs top to bottom offline (`DS_OFFLINE=1`, tiny Laya checkpoint), so none of them rot.

Code cells run in one namespace; `%%writefile` cells write their file; other `!`/`%` lines are skipped.
Notebooks whose metadata has `"decisionsmith": {"offline": false}` (GPU ones) are skipped, and so are notebooks whose
`"imports"` (e.g. a framework) are not installed in this environment.
"""

import json
import os
from pathlib import Path

import pytest

from tests.test_examples import KEYS

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def cells(path):
    nb = json.loads(path.read_text())
    return nb.get("metadata", {}).get("decisionsmith", {}), [c for c in nb["cells"] if c["cell_type"] == "code"]


def run_cell(source, namespace):
    lines = source.splitlines()
    if lines and lines[0].startswith("%%writefile"):
        Path(lines[0].split(maxsplit=1)[1].strip()).write_text("\n".join(lines[1:]) + "\n")
        return
    if lines and lines[0].startswith("%%"):
        return
    code = "\n".join(line for line in lines if not line.lstrip().startswith(("!", "%")))
    exec(compile(code, "<cell>", "exec"), namespace)


@pytest.mark.parametrize("path", sorted(NOTEBOOKS.glob("*.ipynb")), ids=lambda p: p.name)
def test_notebook_runs_offline(path, tiny, tmp_path, monkeypatch, no_network):
    meta, code = cells(path)
    if meta.get("offline") is False:
        pytest.skip(meta.get("why", "needs a GPU"))
    for module in meta.get("imports", []):
        pytest.importorskip(module)
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
    assert (tmp_path / "a.py").read_text() == "X = 1\n" and ns["y"] == 2
