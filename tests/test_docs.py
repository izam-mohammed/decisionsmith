"""Every ```python block in README.md and docs/*.md runs offline, so the docs can't drift from the code.

Blocks on one page run in order in one namespace (a later block may use a class an earlier one defined). They run in
a folder holding the files the docs name: `tickets.csv` and `test.csv` (columns `text`, `label`, `team`,
`wants_refund`), `texts.txt`, `app.py` with the `Ticket` class and `decisions.db` (a shadow-mode harness log of 60
decisions). A block that can't run offline carries
`<!-- no-test: reason -->` on the line before it.
"""

import csv
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BLOCK = re.compile(r"(?:<!--\s*no-test:\s*(?P<reason>[^>]*?)\s*-->\s*\n)?```python\n(?P<code>.*?)```", re.S)
SKIP_PAGES = {
    "roadmap.md": "local planning notes, not part of the docs",
    "finetune.md": "rewritten in the notebooks pull request, which adds its test markers",
}
APP = """from typing import Annotated, Literal

from pydantic import BaseModel, Field

import decisionsmith as ds


class Ticket(BaseModel):
    team: Annotated[
        Literal["billing", "technical", "sales"],
        ds.Options(billing="payments, refunds", technical="bugs, outages", sales="pricing"),
    ]
    wants_refund: bool = Field(description="Does the customer ask for their money back?")
"""


def pages():
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        if path.name in SKIP_PAGES and path.parent.name == "docs":
            continue
        blocks = [(m.group("code"), m.group("reason")) for m in BLOCK.finditer(path.read_text(encoding="utf-8"))]
        if blocks:
            yield pytest.param(path, blocks, id=str(path.relative_to(ROOT)).replace("\\", "/"))


def fixtures(folder: Path) -> None:
    rows = list(csv.DictReader((ROOT / "examples" / "data" / "toy.csv").open(encoding="utf-8")))
    for name, part in (("tickets.csv", rows[:220]), ("test.csv", rows[220:])):
        with open(folder / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["text", "label", "team", "wants_refund"])
            w.writeheader()
            w.writerows(
                {"text": r["text"], "label": r["team"], "team": r["team"], "wants_refund": r["wants_refund"]}
                for r in part
            )
    (folder / "texts.txt").write_text("\n".join(r["text"] for r in rows[:60]), encoding="utf-8")
    (folder / "app.py").write_text(APP, encoding="utf-8")
    import decisionsmith as ds

    labels = ["billing", "technical", "sales"]
    with ds.harness(ds.model(labels), teacher="claude-haiku-4-5", mode="shadow", log=str(folder / "decisions.db")) as h:
        h.many([r["text"] for r in rows[:60]])


def test_every_block_is_tested_or_says_why():
    for path, blocks in (p.values for p in pages()):
        for code, reason in blocks:
            assert reason is None or reason.strip(), "%s: a no-test marker needs a reason" % path.name


@pytest.mark.parametrize("path,blocks", list(pages()))
def test_docs_blocks_run(path, blocks, tiny, tmp_path, monkeypatch, no_network, capsys):
    from tests.test_examples import KEYS

    monkeypatch.setenv("DS_OFFLINE", "1")
    monkeypatch.setenv("DS_LAYA", str(tiny))
    for key in KEYS:
        monkeypatch.setenv(key, os.environ.get(key) or "offline")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "app", raising=False)
    fixtures(tmp_path)
    capsys.readouterr()
    namespace: dict = {"__name__": "__docs__"}
    ran = 0
    for i, (code, reason) in enumerate(blocks):
        if reason:
            continue
        try:
            exec(compile(code, "%s block %d" % (path.name, i + 1), "exec"), namespace)
        except Exception as e:
            pytest.fail("%s block %d failed: %s: %s\n%s" % (path.name, i + 1, type(e).__name__, e, code))
        ran += 1
    for obj in namespace.values():
        close = getattr(obj, "close", None)
        if type(obj).__name__ == "Harness" and callable(close):
            close()
    if not ran:
        pytest.skip("every block on this page is marked no-test")
