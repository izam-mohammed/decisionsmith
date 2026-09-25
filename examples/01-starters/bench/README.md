# Bench engines

One table comparing an LLM and Laya on the same labelled CSV.

```python
"""Compare engines on labelled data: accuracy, calibration, latency and cost per field."""

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


print(ds.bench(Ticket, TOY, ["claude-haiku-4-5", "laya"]))
```

## Run

```bash
pip install "decisionsmith[laya,anthropic]"
python examples/01-starters/bench/main.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
