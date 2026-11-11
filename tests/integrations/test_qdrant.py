import pytest

pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient, models

import decisionsmith as ds
from decisionsmith.integrations.qdrant import filter_points, points, tag
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, truth

TEXTS = ["refund my card charge", "sync is broken", "how much is the plan"]
VECTORS = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def judge():
    def relevant(text):
        return {"label": "relevant" if "refund" in text.split("Document: ")[-1] else "off-topic"}

    return ds.model(["relevant", "off-topic"], FakeEngine(relevant))


@pytest.fixture
def client():
    c = QdrantClient(":memory:")
    c.create_collection("tickets", vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE))
    return c


def test_ingest_tags_then_filter_in_the_query(client):
    h = ds.harness(Ticket, teacher=FakeEngine(truth, confidence=1.0), log=None)
    client.upsert("tickets", points=points(h, [1, 2, 3], VECTORS, TEXTS, [{"n": 1}, None, None]))
    only = models.Filter(must=[models.FieldCondition(key="team", match=models.MatchValue(value="technical"))])
    hits = client.query_points("tickets", query=[0.5, 0.5, 0.5], query_filter=only, limit=3).points
    assert [p.id for p in hits] == [2] and hits[0].payload["team_source"] == "teacher"
    first = client.retrieve("tickets", [1])[0].payload
    assert first == {
        "text": "refund my card charge",
        "n": 1,
        "team": "billing",
        "team_confidence": 1.0,
        "team_source": "teacher",
        "wants_refund": True,
        "wants_refund_confidence": 1.0,
        "wants_refund_source": "teacher",
    }


def test_tag_prefix_fields_and_empty_texts():
    m = ds.model(["billing", "technical", "sales"], FakeEngine(lambda t: {"label": truth(t)["team"]}))
    assert tag(m, ["", "sync is broken"], prefix="ds_")[0] == {
        "ds_label": None,
        "ds_label_confidence": None,
        "ds_label_source": None,
    }


def test_filter_retrieved_points(client):
    client.upsert("tickets", points=points(judge(), [1, 2, 3], VECTORS, TEXTS, text_key="body"))
    res = client.query_points("tickets", query=[0.6, 0.4, 0.3], limit=3)
    kept = filter_points(judge(), res, query="how do refunds work", text_key="body")
    assert [p.id for p in kept] == [1]
    scrolled, _ = client.scroll("tickets")
    assert [p.id for p in filter_points(judge(), scrolled, text_key="body")] == [1]
    assert filter_points(judge(), [models.Record(id=9, payload=None)]) == []
    h = ds.harness(Ticket, teacher=FakeEngine(truth), log=None)
    assert [p.id for p in filter_points(h, scrolled, "team", keep=["sales"], text_key="body")] == [3]
    with pytest.raises(ValueError, match="pass field="):
        filter_points(h, scrolled)
