"""Instructor with a decisionsmith model in front: `create(response_model=Ticket)` is answered by the model when it
is sure, and by the LLM through Instructor otherwise (OPENAI_API_KEY).

Any Instructor client works the same way (`instructor.from_provider("anthropic/claude-haiku-4-5")`, ...).
"""

import instructor
from _schema import TEXTS, Ticket
from openai import OpenAI

import decisionsmith as ds
from decisionsmith.integrations.instructor import wrap

client = wrap(instructor.from_openai(OpenAI()), ds.model(Ticket))

for text in TEXTS[:3]:
    ticket = client.create(model="gpt-5-mini", response_model=Ticket, messages=[{"role": "user", "content": text}])
    print(ticket, "<-", text)
