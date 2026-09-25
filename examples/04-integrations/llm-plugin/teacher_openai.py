"""Any `llm` model as the teacher: keys and plugins come from your llm setup (`llm keys set openai`).

Models from llm plugins work the same way (`llm install llm-anthropic`, then `llm.get_model("claude-haiku-4.5")`).
"""

import llm
from _schema import TEXTS, Ticket

import decisionsmith as ds

h = ds.harness(Ticket, teacher=llm.get_model("gpt-5-mini"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
