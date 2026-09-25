"""Compare engines on labelled data. DS_ENGINES=claude-haiku-4-5,jev,laya python examples/03_bench.py"""

import os

from _common import TOY
from schema import Ticket

import decisionsmith as ds

print(ds.bench(Ticket, TOY, os.environ.get("DS_ENGINES", "fake,laya")))
