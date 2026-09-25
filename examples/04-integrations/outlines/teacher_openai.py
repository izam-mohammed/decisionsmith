"""An OpenAI model through Outlines (OPENAI_API_KEY): Outlines sends the answer schema as structured output."""

import openai
import outlines
from _schema import TEXTS, Ticket

import decisionsmith as ds
from decisionsmith.integrations.outlines import teacher

model = ds.model(Ticket)
rows = model.label(TEXTS, teacher=teacher(outlines.from_openai(openai.OpenAI(), "gpt-5-mini")))
print(len(rows), "rows labelled")
