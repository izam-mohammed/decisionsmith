"""An actor pool: each actor builds the harness once (`fn_constructor_args=(build,)`) and decides many batches.

Use this for a harness (its SQLite log can't be pickled) or a Laya model you want loaded once per actor.
"""

from typing import Annotated, Literal

import ray
from pydantic import BaseModel, Field

import decisionsmith as ds
from decisionsmith.integrations.ray import Decide


class Ticket(BaseModel):
    team: Annotated[
        Literal["billing", "technical", "sales"],
        ds.Options(billing="payments, invoices, refunds", technical="bugs, outages", sales="pricing, plans"),
    ] = Field(description="Which team should handle this ticket?")
    wants_refund: bool = Field(description="Does the customer ask for their money back?")


def build():
    return ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya", log="decisions.db")


ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
texts = ["I was charged twice, please refund me", "The app crashes on login", "Can I get a quote?"] * 4
data = ray.data.from_items([{"id": i, "text": t} for i, t in enumerate(texts)])
out = data.map_batches(Decide, fn_constructor_args=(build,), concurrency=2, batch_size=4, batch_format="pandas")
df = out.to_pandas().sort_values("id")
print(df[["text", "team", "team_source", "wants_refund"]].drop_duplicates("text").to_string(index=False))
print(df.groupby("team").size().to_dict())
ray.shutdown()
