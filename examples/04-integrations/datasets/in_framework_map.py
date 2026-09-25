"""`dataset.map(ds_map(x), batched=True)`: decision columns for a Hugging Face dataset, batched."""

from datasets import Dataset

import decisionsmith as ds
from decisionsmith.integrations.datasets import ds_map

data = Dataset.from_dict({"text": ["You charged me twice", "The app keeps crashing", "How much is the team plan?"]})
model = ds.model(["billing", "technical", "sales"])  # base Laya; train it on your data for real use
data = data.map(ds_map(model), batched=True)
print(data.select_columns(["text", "label", "label_confidence"]).to_list())
