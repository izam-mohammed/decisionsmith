# pandas

`df.ds.decide("text", x)`: a column per field plus `<field>_confidence` and `<field>_source`, batched; or `decide(df, "text", x, prefix="pred_")`.

```python
"""`df.ds.decide(...)`: a label column (plus confidence and source) for every text in a DataFrame, batched.

Importing `decisionsmith.integrations.pandas` adds the `.ds` accessor to every DataFrame.
"""

import pandas as pd

import decisionsmith as ds
import decisionsmith.integrations.pandas

df = pd.DataFrame({"text": ["You charged me twice", "The app keeps crashing", "How much is the team plan?", None]})
model = ds.model(["billing", "technical", "sales"])  # base Laya; train it on your data for real use
print(df.ds.decide("text", model))
```

| file | what it shows |
|---|---|
| [`in_framework_accessor.py`](in_framework_accessor.py) | `df.ds.decide(...)`: a label column (plus confidence and source) for every text in a DataFrame, batched. |
| [`in_framework_evaluate_labels.py`](in_framework_evaluate_labels.py) | Check a harness against labelled rows: decide into `pred_` columns, compare, and list the unsure ones to review. |

## Run

```bash
uv add "decisionsmith[pandas,laya]"
uv run python examples/04-integrations/pandas/in_framework_accessor.py
uv run python examples/04-integrations/pandas/in_framework_evaluate_labels.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
