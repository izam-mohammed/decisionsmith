"""Jev as the teacher, Laya as the student. Needs TYPESAFE_API_KEY (each call is billed by TypeSafe)."""

import os
import sys

from _common import toy_rows
from schema import Ticket

import decisionsmith as ds

if not os.environ.get("TYPESAFE_API_KEY"):
    sys.exit("set TYPESAFE_API_KEY to run this example")

h = ds.harness(Ticket, teacher="jev", student="laya", mode="shadow")
h.many([r["text"] for r in toy_rows()[:50]])
print(h.status())
