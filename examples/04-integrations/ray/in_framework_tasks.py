"""`Dataset.map_batches(Decide(model))`: decision columns for a Ray Dataset, one batch per task."""

import ray

import decisionsmith as ds
from decisionsmith.integrations.ray import Decide

ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
texts = ["You charged me twice", "The app keeps crashing", "How much is the team plan?"]
data = ray.data.from_items([{"text": t} for t in texts])
model = ds.model(["billing", "technical", "sales"])  # base Laya; train it on your data for real use
for row in data.map_batches(Decide(model)).take_all():
    print(row["text"], "->", row["label"], round(row["label_confidence"], 2))
ray.shutdown()
