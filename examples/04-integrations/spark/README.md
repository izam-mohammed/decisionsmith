# Spark

`udf(build, field)`: a `pandas_udf` over a text column (one field, or a struct of every field with confidence and source); `build` makes the model or harness in each Python worker. Needs Java 17+.

```python
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
```

| file | what it shows |
|---|---|
| [`in_framework_one_field.py`](in_framework_one_field.py) | `udf(build, field)`: a Spark `pandas_udf` that decides a text column, one Arrow batch at a time. |
| [`in_framework_struct.py`](in_framework_struct.py) | Every field of a harness decision on a Spark DataFrame (a struct you expand with `d.*`), written to Parquet. |

## Run

```bash
uv add "decisionsmith[spark,laya]"
uv run python examples/04-integrations/spark/in_framework_one_field.py
uv run python examples/04-integrations/spark/in_framework_struct.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
