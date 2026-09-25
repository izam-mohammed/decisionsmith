"""`decide(ddf, "text", x)`: decision columns for a Dask DataFrame, one batch per partition, computed lazily."""

import dask.dataframe as dd
import pandas as pd

import decisionsmith as ds
from decisionsmith.integrations.dask import decide

texts = ["You charged me twice", "The app keeps crashing", "How much is the team plan?", "Refund me please"]
ddf = dd.from_pandas(pd.DataFrame({"text": texts}), npartitions=2)
model = ds.model(["billing", "technical", "sales"])  # base Laya; train it on your data for real use
print(decide(ddf, "text", model).compute())
