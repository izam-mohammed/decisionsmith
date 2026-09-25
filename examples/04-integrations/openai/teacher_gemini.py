"""Gemini through its OpenAI-compatible endpoint (GEMINI_API_KEY)."""

import os

from _schema import TEXTS, Ticket
from openai import OpenAI

import decisionsmith as ds
from decisionsmith.integrations.openai import teacher

url = "https://generativelanguage.googleapis.com/v1beta/openai/"
client = OpenAI(base_url=url, api_key=os.environ["GEMINI_API_KEY"])
h = ds.harness(Ticket, teacher=teacher(client, "gemini-2.5-flash"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
