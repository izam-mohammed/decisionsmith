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
