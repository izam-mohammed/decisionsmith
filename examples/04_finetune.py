"""Fine-tune Laya on labelled data and read the report. Head-only on this small set: about a minute on a laptop."""

import os

from _common import TOY
from schema import Ticket

import decisionsmith as ds

report = ds.finetune(TOY, Ticket, base=os.environ.get("DS_BASE", "laya"), out=os.environ.get("DS_OUT", "runs/toy"))
print(report)
print('use it: ds.harness(Ticket, teacher=..., student="laya:%s")' % report.path)
