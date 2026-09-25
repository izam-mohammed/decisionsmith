# Local only (Ollama + Laya)

An Ollama teacher and a Laya student; nothing leaves your machine.

```python
"""Fully local: an Ollama model teaches, Laya learns. No data leaves the machine.

ollama pull qwen3 && python main.py
"""

import csv
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

import decisionsmith as ds

TOY = Path(__file__).parents[2] / "data" / "toy.csv"


class Ticket(BaseModel):
    team: Annotated[
        Literal["billing", "technical", "sales"],
        ds.Options(billing="payments, invoices, refunds", technical="bugs, outages", sales="pricing, plans"),
    ] = Field(description="Which team should handle this ticket?")
    wants_refund: bool = Field(description="Does the customer ask for their money back?")


texts = [row["text"] for row in csv.DictReader(TOY.read_text().splitlines())][:20]

h = ds.harness(Ticket, teacher="ollama/qwen3", student="laya", mode="shadow")
for text in texts:
    print(h.decide(text).value)
print(h.status())
```

## Run

```bash
uv add "decisionsmith[laya]"
uv run python examples/01-starters/local-only/main.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
