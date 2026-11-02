"""Check a harness against labelled rows: decide into `pred_` columns, compare, and list the unsure ones to review.

The same DataFrame flow works for a nightly batch job: read, decide, write back.
"""

import pandas as pd
from _schema import ROWS, Ticket

import decisionsmith as ds
from decisionsmith.integrations.pandas import decide

df = pd.DataFrame(ROWS, columns=["text", "team", "wants_refund"])
h = ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya", log="decisions.db")

out = decide(df, "text", h, prefix="pred_", batch_size=64)
labelled = out.dropna(subset=["team"])
print("team accuracy on %d labelled rows: %.2f" % (len(labelled), (labelled.pred_team == labelled.team).mean()))
print("answered by:", labelled.pred_team_source.value_counts().to_dict())
unsure = labelled[labelled.pred_team_confidence < 0.8]
print("to review:\n", unsure[["text", "team", "pred_team", "pred_team_confidence"]].to_string(index=False))
out.to_csv("tickets_decided.csv", index=False)
