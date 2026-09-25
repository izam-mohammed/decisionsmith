"""Start with a teacher, measure the student in shadow mode, calibrate it, see where each field stands.

python examples/01_quickstart.py                      # stand-in teacher, real Laya student
DS_TEACHER=claude-haiku-4-5 python examples/01_quickstart.py
"""

import os

from _common import teacher, toy_rows
from schema import Ticket

import decisionsmith as ds

h = ds.harness(
    Ticket,
    teacher=teacher(),
    student=os.environ.get("DS_STUDENT", "laya"),
    mode="shadow",
    log=os.environ.get("DS_LOG", "decisions.db"),
)

texts = [r["text"] for r in toy_rows()]
print(h(texts[2]), "<-", texts[2])
h.many(texts)
print(h.status())
print(h.adapt())
