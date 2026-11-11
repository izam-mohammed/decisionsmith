# Polars

`decide_expr(x, "text", field)`: a decision as a Polars expression (one field, or a struct of every field with confidence and source), batched through `map_batches`; works on lazy frames.

```python
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
```

| file | what it shows |
|---|---|
| [`in_framework_lazy_scan.py`](in_framework_lazy_scan.py) | A lazy pipeline: scan a CSV, decide every field (a struct you unnest), keep the sure rows, write Parquet. |
| [`in_framework_one_field.py`](in_framework_one_field.py) | `decide_expr(x, "text", field)`: one decision field as a Polars column, batched through `map_batches`. |

## Run

```bash
uv add "decisionsmith[polars,laya]"
uv run python examples/04-integrations/polars/in_framework_lazy_scan.py
uv run python examples/04-integrations/polars/in_framework_one_field.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
