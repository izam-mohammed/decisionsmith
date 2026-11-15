"""`df.ds.decide(...)`: a label column (plus confidence and source) for every text in a DataFrame, batched.

Importing `decisionsmith.integrations.pandas` adds the `.ds` accessor to every DataFrame.
"""

import pandas as pd

import decisionsmith as ds
import decisionsmith.integrations.pandas

df = pd.DataFrame({"text": ["You charged me twice", "The app keeps crashing", "How much is the team plan?", None]})
model = ds.model(["billing", "technical", "sales"])  # base Laya; train it on your data for real use
print(df.ds.decide("text", model))
