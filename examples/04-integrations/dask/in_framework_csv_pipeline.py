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
