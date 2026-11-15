import pytest

pytest.importorskip("datasets")

import datasets

import decisionsmith as ds
from decisionsmith.integrations.datasets import ds_map, records
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, corpus, truth


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", str(tmp_path / "hf-cache"))


def test_map_batched_and_not_batched(db):
    h = ds.harness(Ticket, teacher=FakeEngine(truth, confidence=1.0), log=db)
    d = datasets.Dataset.from_dict({"text": ["refund my card charge", None, "sync is broken"], "id": [1, 2, 3]})
    out = d.map(ds_map(h), batched=True, batch_size=2)
    assert out.column_names == [
        "text",
        "id",
        "team",
        "team_confidence",
        "team_source",
        "wants_refund",
        "wants_refund_confidence",
        "wants_refund_source",
    ]
    assert out["team"] == ["billing", None, "technical"] and out["wants_refund"] == [True, None, False]
    assert out["team_source"] == ["teacher", None, "teacher"]
    m = ds.model(["billing", "technical", "sales"], FakeEngine(lambda t: {"label": truth(t)["team"]}))
    one = d.map(ds_map(m, prefix="p_"))
    assert one["p_label"] == ["billing", None, "technical"] and one["p_label_source"][0] == "student"
    two = d.map(ds_map(h, fields="team", batch_size=1), batched=True)
    assert "wants_refund" not in two.column_names and two["team"][2] == "technical"


def _folder(root):
    feats = datasets.Features(
        {
            "text": datasets.Value("string"),
            "team": datasets.ClassLabel(names=["billing", "technical", "sales"]),
            "wants_refund": datasets.Value("bool"),
        }
    )
    texts = corpus(60)
    rows = {
        "text": texts,
        "team": [["billing", "technical", "sales"].index(truth(t)["team"]) for t in texts],
        "wants_refund": [truth(t)["wants_refund"] for t in texts],
    }
    rows["team"][0] = -1
    (root / "tickets").mkdir()
    datasets.Dataset.from_dict(rows, features=feats).to_parquet(str(root / "tickets" / "train.parquet"))
    datasets.Dataset.from_dict({k: v[:6] for k, v in rows.items()}, features=feats).to_parquet(
        str(root / "tickets" / "test.parquet")
    )


def test_records_from_a_local_folder(tmp_path, monkeypatch):
    _folder(tmp_path)
    monkeypatch.chdir(tmp_path)
    rows = list(records("tickets"))
    assert len(rows) == 60 and rows[0][0] == "hf:tickets:train row 0"
    assert rows[0][1]["team"] is None and rows[1][1]["team"] == truth(rows[1][1]["text"])["team"]
    assert len(list(records("tickets:test"))) == 6 and len(list(records("tickets:train[:10%]"))) == 6
    with pytest.raises(ValueError, match="hf:<dataset>:<split>"):
        list(records("tickets:bad split"))


def test_train_and_evaluate_from_hf(tiny, tmp_path, monkeypatch):
    _folder(tmp_path)
    monkeypatch.chdir(tmp_path)
    rep = ds.finetune(
        "hf:tickets", Ticket, base=str(tiny), out=str(tmp_path / "run"), epochs=1, device="cpu", verbose=False
    )
    assert rep.details["provenance"]["rows"]["train"] > 0
    m = ds.model(Ticket, str(tiny))
    assert m.evaluate("hf:tickets:test").rows
