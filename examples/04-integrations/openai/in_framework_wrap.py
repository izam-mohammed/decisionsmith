"""Wrap an OpenAI client: requests whose `response_format` only asks for decisions are answered by your model
when it is sure; everything else, and anything it is unsure about, goes to the API as before (OPENAI_API_KEY).

Train the model first (`model.train(...)`) so it is sure more often; base Laya mostly falls through.
"""

from _schema import TEXTS, Ticket
from openai import OpenAI

import decisionsmith as ds
from decisionsmith.integrations.openai import wrap

client = wrap(OpenAI(), ds.model(Ticket))

for text in TEXTS[:3]:
    r = client.chat.completions.parse(
        model="gpt-5-mini",
        messages=[{"role": "system", "content": "Route the support ticket."}, {"role": "user", "content": text}],
        response_format=Ticket,
    )
    who = "decisionsmith" if r.id == "decisionsmith" else "the API"
    print(r.choices[0].message.parsed, "<- answered by", who)
