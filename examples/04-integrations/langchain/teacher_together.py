"""Together AI through ChatTogether (TOGETHER_API_KEY)."""

from _schema import TEXTS, Ticket
from langchain_together import ChatTogether

import decisionsmith as ds

model = ds.model(Ticket)
rows = model.label(TEXTS, teacher=ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo"))
print(len(rows), "rows labelled; train Laya on them with model.train(rows)")
