import os
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyspark")
if not (os.environ.get("JAVA_HOME") or shutil.which("java")):
    pytest.skip("Spark needs Java 17+ (JAVA_HOME or java on PATH)", allow_module_level=True)

import pandas as pd

import decisionsmith as ds
from decisionsmith.integrations.spark import udf
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, truth

ROOT = str(Path(__file__).resolve().parents[2])


def build():
    return ds.harness(Ticket, teacher=FakeEngine(truth, confidence=1.0), log=None)


@pytest.fixture(scope="module")
def spark():
    from pyspark import SparkContext
    from pyspark.sql import SparkSession

    env = {"PYSPARK_PYTHON": sys.executable, "SPARK_LOCAL_IP": "127.0.0.1", "PYTHONPATH": ROOT}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    s = (
        SparkSession.builder.master("local[2]")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    yield s
    s.stop()
    SparkContext._gateway.shutdown()  # the next session (the examples) starts a JVM with their environment
    SparkContext._gateway = SparkContext._jvm = None
    for k, v in old.items():
        if v is None:
            os.environ.pop(k)
        else:
            os.environ[k] = v


def frame(spark):
    return spark.createDataFrame([("refund my card charge",), (None,), ("sync is broken",)], "text string")


def test_one_field_on_a_real_spark_session(spark):
    rows = frame(spark).withColumn("team", udf(build, "team")("text")).collect()
    assert [r.team for r in rows] == ["billing", None, "technical"]
    rows = frame(spark).withColumn("r", udf(build, "wants_refund")("text")).collect()
    assert [r.r for r in rows] == [True, None, False]


def test_every_field_as_a_struct(spark):
    out = frame(spark).withColumn("d", udf(build, prefix="p_")("text")).select("text", "d.*")
    assert out.columns == [
        "text",
        "p_team",
        "p_team_confidence",
        "p_team_source",
        "p_wants_refund",
        "p_wants_refund_confidence",
        "p_wants_refund_source",
    ]
    first = out.collect()[0]
    assert (first.p_team, first.p_team_confidence, first.p_team_source) == ("billing", 1.0, "teacher")
    m = ds.model(["billing", "technical", "sales"], FakeEngine(lambda t: {"label": truth(t)["team"]}))
    assert [r.d.label for r in frame(spark).withColumn("d", udf(m)("text")).collect()] == ["billing", None, "technical"]


def test_the_udf_body_directly():
    one = udf(build, "team").func(pd.Series(["you charged me twice", None]))
    assert one.tolist() == ["billing", None]
    many = udf(build, batch_size=1).func(pd.Series(["sync is broken"]))
    assert many["team_source"].tolist() == ["teacher"] and many["wants_refund"].tolist() == [False]
