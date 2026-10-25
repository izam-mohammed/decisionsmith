"""Every field of a harness decision on a Spark DataFrame (a struct you expand with `d.*`), written to Parquet.

The harness is built inside each Python worker, so its SQLite log stays local to the worker; point `log=` at a
shared path (or `log=None`) on a real cluster.
"""

import os
import sys
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import decisionsmith as ds
from decisionsmith.integrations.spark import udf


class Ticket(BaseModel):
    team: Annotated[
        Literal["billing", "technical", "sales"],
        ds.Options(billing="payments, invoices, refunds", technical="bugs, outages", sales="pricing, plans"),
    ] = Field(description="Which team should handle this ticket?")
    wants_refund: bool = Field(description="Does the customer ask for their money back?")


def build():
    return ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya", log="decisions.db")


os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
spark = SparkSession.builder.master("local[2]").config("spark.ui.enabled", "false").getOrCreate()
texts = ["I was charged twice, please refund me", "The app crashes on login", "Can I get a quote?", None]
df = spark.createDataFrame([(t,) for t in texts], "text string")

decided = df.withColumn("d", udf(build)("text")).select("text", "d.*")
decided.write.mode("overwrite").parquet("tickets_decided")
spark.read.parquet("tickets_decided").groupBy("team").agg(F.count("*").alias("tickets")).show()
decided.where(F.col("team_confidence") < 0.8).select("text", "team", "team_source").show(truncate=False)
spark.stop()
