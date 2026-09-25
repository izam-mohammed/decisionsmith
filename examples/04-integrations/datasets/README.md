# Hugging Face datasets

`dataset.map(ds_map(x), batched=True)` adds decision columns; `model.train("hf:<dataset>")` / `ds.finetune("hf:<dataset>:<split>", ...)` train on a hub dataset or a local folder of data files.

```python
"""`dataset.map(ds_map(x), batched=True)`: decision columns for a Hugging Face dataset, batched."""

from datasets import Dataset

import decisionsmith as ds
from decisionsmith.integrations.datasets import ds_map

data = Dataset.from_dict({"text": ["You charged me twice", "The app keeps crashing", "How much is the team plan?"]})
model = ds.model(["billing", "technical", "sales"])  # base Laya; train it on your data for real use
data = data.map(ds_map(model), batched=True)
print(data.select_columns(["text", "label", "label_confidence"]).to_list())
```

| file | what it shows |
|---|---|
| [`in_framework_map.py`](in_framework_map.py) | `dataset.map(ds_map(x), batched=True)`: decision columns for a Hugging Face dataset, batched. |
| [`train_from_hf.py`](train_from_hf.py) | Train and evaluate on a Hugging Face dataset by name: `model.train("hf:<dataset>")`, `evaluate("hf:<dataset>:test")`. |

## Run

```bash
uv add "decisionsmith[datasets,laya]"
uv run python examples/04-integrations/datasets/in_framework_map.py
uv run python examples/04-integrations/datasets/train_from_hf.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
