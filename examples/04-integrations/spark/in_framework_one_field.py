"""`udf(build, field)`: a Spark `pandas_udf` that decides a text column, one Arrow batch at a time.

Spark needs Java 17+. `build` runs once on the driver (for the schema) and once in each Python worker.
"""

import os
import sys

from pyspark.sql import SparkSession

import decisionsmith as ds
from decisionsmith.integrations.spark import udf

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)  # workers use this Python (with decisionsmith installed)
spark = SparkSession.builder.master("local[2]").config("spark.ui.enabled", "false").getOrCreate()


def build():
    return ds.model(["billing", "technical", "sales"])  # base Laya; train it on your data for real use


df = spark.createDataFrame([("You charged me twice",), ("The app keeps crashing",)], "text string")
df.withColumn("team", udf(build, "label")("text")).show(truncate=False)
spark.stop()
