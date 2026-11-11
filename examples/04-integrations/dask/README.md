# Dask

`decide(ddf, "text", x)`: decision columns for a Dask DataFrame via `map_partitions`, lazy; `x` can be a function that builds the model or harness per worker.

```python
"""A batch job over many CSV files: read them as one Dask DataFrame, decide, write Parquet, count per team.

`build` makes the harness; with the process scheduler (or dask.distributed) each worker process builds its own,
since a harness (its SQLite log) can't be pickled.
"""

import csv

import dask.dataframe as dd
from _schema import ROWS, Ticket

import decisionsmith as ds
from decisionsmith.integrations.dask import decide

for part in range(3):
    with open("tickets-%d.csv" % part, "w", newline="") as f:
        csv.writer(f).writerows([("id", "text"), *[(part * 100 + i, r[0]) for i, r in enumerate(ROWS) if r[0]]])


def build():
    return ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya", log="decisions.db")


out = decide(dd.read_csv("tickets-*.csv"), "text", build, ["team", "wants_refund"])
out.to_parquet("tickets_decided", write_index=False)
print(dd.read_parquet("tickets_decided").groupby("team").size().compute().to_dict())
```

| file | what it shows |
|---|---|
| [`in_framework_csv_pipeline.py`](in_framework_csv_pipeline.py) | A batch job over many CSV files: read them as one Dask DataFrame, decide, write Parquet, count per team. |
| [`in_framework_decide.py`](in_framework_decide.py) | `decide(ddf, "text", x)`: decision columns for a Dask DataFrame, one batch per partition, computed lazily. |

## Run

```bash
uv add "decisionsmith[dask,laya]"
uv run python examples/04-integrations/dask/in_framework_csv_pipeline.py
uv run python examples/04-integrations/dask/in_framework_decide.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
