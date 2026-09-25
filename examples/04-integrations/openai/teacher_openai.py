"""OpenAI through the official SDK (OPENAI_API_KEY)."""

from _schema import TEXTS, Ticket
from openai import OpenAI

import decisionsmith as ds
from decisionsmith.integrations.openai import teacher

client = OpenAI()
h = ds.harness(Ticket, teacher=teacher(client, "gpt-5-mini"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
