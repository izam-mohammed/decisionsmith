import uuid

import pytest

pytest.importorskip("chromadb")

import chromadb

import decisionsmith as ds
from decisionsmith.integrations.chroma import add, filter_results, tag
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, truth

DOCS = ["refund my card charge", "sync is broken", "how much is the plan"]
VECTORS = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def judge():
    def relevant(text):
        return {"label": "relevant" if "refund" in text.split("Document: ")[-1] else "off-topic"}

    return ds.model(["relevant", "off-topic"], FakeEngine(relevant))


@pytest.fixture
def collection():
    client = chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
    name = "test-%s" % uuid.uuid4().hex
    yield client.create_collection(name, embedding_function=None)
    client.delete_collection(name)


def test_ingest_tags_then_filter_with_where(collection):
    h = ds.harness(Ticket, teacher=FakeEngine(truth, confidence=1.0), log=None)
    add(collection, h, ids=["1", "2", "3"], documents=DOCS, embeddings=VECTORS, metadatas=[{"n": 1}, None, {"n": 3}])
    add(collection, h, ids=["4"], documents=[""], embeddings=[[0.5, 0.5, 0.5]])
    assert collection.get(ids=["4"])["metadatas"] == [None]
    got = collection.get(where={"team": "billing"})
    assert got["documents"] == ["refund my card charge"]
    assert got["metadatas"][0] == {
        "n": 1,
        "team": "billing",
        "team_confidence": 1.0,
        "team_source": "teacher",
        "wants_refund": True,
        "wants_refund_confidence": 1.0,
        "wants_refund_source": "teacher",
    }
    hits = collection.query(query_embeddings=[[0.9, 0.1, 0.1]], n_results=3, where={"wants_refund": False})
    assert hits["documents"] == [["sync is broken", "how much is the plan"]]


def test_tag_skips_empty_documents_and_prefixes():
    m = ds.model(["billing", "technical", "sales"], FakeEngine(lambda t: {"label": truth(t)["team"]}))
    assert tag(m, ["sync is broken", ""], fields="label", prefix="ds_") == [
        {"ds_label": "technical", "ds_label_confidence": 0.9, "ds_label_source": "student"},
        None,
    ]


def test_filter_query_results_after_retrieval(collection):
    collection.add(ids=["1", "2", "3"], documents=DOCS, embeddings=VECTORS)
    hits = collection.query(
        query_embeddings=[[1.0, 0.2, 0.2], [0.2, 1.0, 0.2]],
        n_results=3,
        include=["documents", "distances", "embeddings"],
    )
    kept = filter_results(judge(), hits, query=["how do refunds work", "refund?"])
    assert kept["documents"] == [["refund my card charge"], ["refund my card charge"]]
    assert kept["ids"] == [["1"], ["1"]] and len(kept["distances"][0]) == 1 and len(kept["embeddings"][1]) == 1
    assert kept["included"] == hits["included"] and kept["uris"] is None
    flat = filter_results(judge(), collection.get(), query="refunds")
    assert flat["ids"] == ["1"] and flat["documents"] == ["refund my card charge"]
    alone = filter_results(judge(), {"ids": ["1", "2"], "documents": ["refund me", None]})
    assert alone == {"ids": ["1"], "documents": ["refund me"]}
    assert filter_results(judge(), {"ids": [], "documents": []}) == {"ids": [], "documents": []}


def test_filter_on_one_field_of_a_harness():
    h = ds.harness(Ticket, teacher=FakeEngine(truth), log=None)
    res = {"ids": [["1", "2"]], "documents": [["sync is broken", "refund my card charge"]]}
    assert filter_results(h, res, "team", keep=["technical"])["ids"] == [["1"]]
    with pytest.raises(ValueError, match="pass field="):
        filter_results(h, res)
