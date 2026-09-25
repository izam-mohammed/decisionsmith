import math

import pytest

torch = pytest.importorskip("torch")

from decisionsmith.training import checkpoint, laya_compat  # noqa: E402


def test_round_trip_without_pickle(tmp_path):
    p = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.AdamW([p])
    p.grad = torch.ones(3)
    opt.step()
    state = {
        "optimizer": opt.state_dict(),
        "betas": (0.9, 0.999),
        "best": None,
        "best_loss": math.inf,
        "log": [{"epoch": 1, "train_loss": 0.5}],
        "rng": torch.get_rng_state(),
        "flag": True,
    }
    checkpoint.save(str(tmp_path), state)
    assert checkpoint.exists(str(tmp_path))
    back = checkpoint.load(str(tmp_path))
    assert back["betas"] == (0.9, 0.999) and back["best"] is None and back["best_loss"] == math.inf
    assert back["log"] == state["log"] and back["flag"] is True and torch.equal(back["rng"], state["rng"])
    assert torch.equal(back["optimizer"]["state"][0]["exp_avg"], opt.state_dict()["state"][0]["exp_avg"])
    fresh = torch.optim.AdamW([torch.nn.Parameter(torch.ones(3))])
    fresh.load_state_dict(back["optimizer"])


def test_refuses_what_it_cannot_store(tmp_path):
    with pytest.raises(TypeError, match="can't store a set"):
        checkpoint.save(str(tmp_path), {"x": {1, 2}})
    assert not checkpoint.exists(str(tmp_path / "none"))


def test_legacy_pickle_is_refused(tmp_path):
    (tmp_path / "state.pt").write_bytes(b"x")
    assert checkpoint.exists(str(tmp_path))
    with pytest.raises(ValueError, match="pickled files are never loaded"):
        checkpoint.load(str(tmp_path))


def test_laya_question_api_matches_what_training_expects():
    pytest.importorskip("laya")
    choice = laya_compat.internal_question(
        "team", {"type": "choice", "instructions": "Which team?", "criteria": ["a", "b"]}
    )
    assert choice == {"t": "choice", "ins": "Which team?", "crit": {"a": None, "b": None}}
    noul = laya_compat.internal_question("ok", {"type": "noul", "instructions": "Ok?", "criteria": {True: "yes"}})
    assert noul["crit"] == {"true": "yes"}
    score = laya_compat.internal_question("u", {"type": "score", "instructions": "Rate.", "criteria": ["low", "high"]})
    assert score["t"] == "score" and score["crit"] == ["low", "high"]
    with pytest.raises(ValueError, match="unknown type"):
        laya_compat.internal_question("x", {"type": "nope", "instructions": "?"})


def test_laya_api_change_is_one_clear_error(monkeypatch):
    pytest.importorskip("laya")
    from laya.agent import Agent

    monkeypatch.delattr(Agent, "_to_internal")
    with pytest.raises(RuntimeError, match="changed the question API"):
        laya_compat.internal_question("x", {"type": "noul", "instructions": "?"})
