"""SQL functions `ds_decide(text, field)`, `ds_confidence(text, field)` and `ds_is_true(text)` in DuckDB."""

from typing import Literal

import duckdb
from pydantic import BaseModel

import decisionsmith as ds
from decisionsmith.integrations.duckdb import register


class Ticket(BaseModel):
    team: Literal["billing", "technical", "sales"]
    wants_refund: bool


con = duckdb.connect()
register(con, ds.model(Ticket))  # base Laya; train it on your data for real use
print(con.sql("SELECT ds_decide('You charged me twice, refund me', 'team') AS team"))
print(con.sql("SELECT ds_is_true('You charged me twice, refund me') AS wants_refund"))
