"""An OpenAI model through the Agents SDK as the teacher (OPENAI_API_KEY)."""

from _schema import TEXTS, Ticket
from agents import OpenAIResponsesModel
from openai import AsyncOpenAI

import decisionsmith as ds
from decisionsmith.integrations.openai_agents import teacher

model = OpenAIResponsesModel(model="gpt-5-mini", openai_client=AsyncOpenAI())
h = ds.harness(Ticket, teacher=teacher(model), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
