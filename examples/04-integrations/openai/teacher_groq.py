"""Groq through the OpenAI SDK (GROQ_API_KEY)."""

import os

from _schema import TEXTS, Ticket
from openai import OpenAI

import decisionsmith as ds
from decisionsmith.integrations.openai import teacher

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"])
h = ds.harness(Ticket, teacher=teacher(client, "llama-3.3-70b-versatile"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
