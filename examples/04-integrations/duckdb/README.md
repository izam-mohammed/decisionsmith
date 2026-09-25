# DuckDB

SQL functions `ds_decide(text, field)`, `ds_confidence(text, field)` and `ds_is_true(text)` (vectorised Arrow UDFs, each text decided once per session).

```python
"""A SQL report over a CSV: tickets per team, how sure the model is, and the refund requests, straight from DuckDB.

Each text is decided once per session even when a query asks for several fields.
"""

import csv

import duckdb
from _schema import ROWS, Ticket

import decisionsmith as ds
from decisionsmith.integrations.duckdb import register

with open("tickets.csv", "w", newline="") as f:
    csv.writer(f).writerows([("text",), *[(r[0],) for r in ROWS]])

con = duckdb.connect()
register(con, ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya", log="decisions.db"))
con.sql("CREATE TABLE tickets AS SELECT text FROM read_csv('tickets.csv') WHERE text <> ''")
print(
    con.sql(
        """SELECT ds_decide(text, 'team') AS team, count(*) AS tickets,
                  round(avg(ds_confidence(text, 'team')), 2) AS avg_confidence
           FROM tickets GROUP BY team ORDER BY tickets DESC"""
    )
)
print(con.sql("SELECT text FROM tickets WHERE ds_is_true(text)"))
```

| file | what it shows |
|---|---|
| [`in_framework_sql_report.py`](in_framework_sql_report.py) | A SQL report over a CSV: tickets per team, how sure the model is, and the refund requests, straight from DuckDB. |
| [`in_framework_udfs.py`](in_framework_udfs.py) | SQL functions `ds_decide(text, field)`, `ds_confidence(text, field)` and `ds_is_true(text)` in DuckDB. |

## Run

```bash
uv add "decisionsmith[duckdb,laya]"
uv run python examples/04-integrations/duckdb/in_framework_sql_report.py
uv run python examples/04-integrations/duckdb/in_framework_udfs.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
