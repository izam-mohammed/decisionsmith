"""`decide_expr(x, "text", field)`: one decision field as a Polars column, batched through `map_batches`."""

import polars as pl

import decisionsmith as ds
from decisionsmith.integrations.polars import decide_expr

df = pl.DataFrame({"text": ["You charged me twice", "The app keeps crashing", "How much is the team plan?"]})
model = ds.model(["billing", "technical", "sales"])  # base Laya; train it on your data for real use
print(df.with_columns(decide_expr(model, "text", "label").alias("team")))
