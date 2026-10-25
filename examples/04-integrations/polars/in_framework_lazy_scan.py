"""A lazy pipeline: scan a CSV, decide every field (a struct you unnest), keep the sure rows, write Parquet.

Unsure rows go to a review file instead, so a person (or the teacher LLM) can label them.
"""

import polars as pl
from _schema import ROWS, Ticket

import decisionsmith as ds
from decisionsmith.integrations.polars import decide_expr

pl.DataFrame({"text": [r[0] for r in ROWS]}).write_csv("tickets.csv")
h = ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya", log="decisions.db")

decided = (
    pl.scan_csv("tickets.csv")
    .with_columns(decide_expr(h, "text"))
    .unnest("decision")
    .filter(pl.col("text").is_not_null())
    .collect()
)
sure = decided.filter(pl.col("team_confidence") >= 0.8)
sure.write_parquet("tickets_decided.parquet")
decided.filter(pl.col("team_confidence") < 0.8).write_csv("to_review.csv")
print(decided.select("text", "team", "team_source", "wants_refund"))
print("sure: %d of %d" % (sure.height, decided.height))
