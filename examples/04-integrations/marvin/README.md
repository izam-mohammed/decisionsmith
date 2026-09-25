# Marvin classify drop-in

A classify(data, labels) with Marvin's signature, answered by a ds.model (base Laya or your trained one).

```python
"""Marvin's `classify()` answered by Laya instead of an LLM: same call, no API key, runs on your machine.

Swap `from marvin import classify` for the import below; pass `model=` to use a model you trained.
"""

import enum

from decisionsmith.integrations.marvin import classify


class Team(enum.Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    SALES = "sales"


print(classify("I was charged twice this month", ["billing", "technical", "sales"]))
print(classify("The app crashes every time I log in", Team, instructions="Which team should handle this?"))
```

## Run

```bash
uv add "decisionsmith[laya]"
uv run python examples/04-integrations/marvin/in_framework_classify.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
