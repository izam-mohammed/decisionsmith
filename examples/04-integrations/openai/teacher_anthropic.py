"""Claude through Anthropic's OpenAI SDK compatibility endpoint (ANTHROPIC_API_KEY)."""

import os

from _schema import TEXTS, Ticket
from openai import OpenAI

import decisionsmith as ds
from decisionsmith.integrations.openai import teacher

client = OpenAI(base_url="https://api.anthropic.com/v1/", api_key=os.environ["ANTHROPIC_API_KEY"])
h = ds.harness(Ticket, teacher=teacher(client, "claude-haiku-4-5"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
