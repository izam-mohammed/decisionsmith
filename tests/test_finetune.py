import json
import random
from contextlib import nullcontext

import numpy as np
import pytest
import torch

import decisionsmith as ds
from decisionsmith.training import finetuning as fmod
from decisionsmith.training import train as tr
from decisionsmith.training.data import load, split
from tests.conftest import Ticket, corpus, truth


def _weights(path):
    from safetensors.torch import load_file

    return load_file(str(path / "model.safetensors"))


def test_finetune_writes_a_laya_checkpoint(tiny, toy_csv, tmp_path):
    import laya

    out = tmp_path / "run"
    rep = ds.finetune(toy_csv, Ticket, base=str(tiny), out=str(out), epochs=2, device="cpu", verbose=False)
    for name in (
        "model.safetensors",
        "rl_agent_config.json",
        "report.json",
        "report.html",
        "MODEL_CARD.md",
        "train_log.jsonl",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
        "checkpoint_latest/state.safetensors",
        "checkpoint_latest/state.json",
    ):
        assert (out / name).exists(), name
    cfg = json.loads((out / "rl_agent_config.json").read_text())
    assert cfg["fine_tuned"] is True and len(cfg["temperature"]) == 3 and "temperature_by_options" not in cfg
    assert cfg["decisionsmith"]["train"] == "head" and cfg["decisionsmith"]["rows"]["test"] == 9
    assert cfg["decisionsmith"]["schema"]["name"] == "Ticket" and cfg["max_len"] == 64
    agent = laya.load(str(out), device="cpu")
    assert agent.predict("you charged me twice", ds.harness(Ticket, teacher="fake", log=None).schema.questions())
    assert rep.go is False and any("want 100" in r for r in rep.reasons)
    assert rep.rows[-1]["field"] == "all" and rep.path == str(out.resolve())
    assert len(rep.details["train_ids"]) == 45 and rep.details["ms_per_decision"] > 0
    assert rep.details["finetuned"]["worst"][0]["text"]
    assert "base_model: %s" % tiny in (out / "MODEL_CARD.md").read_text()
    assert json.loads((out / "report.json").read_text())["kind"] == "finetune"
    lines = (out / "train_log.jsonl").read_text().splitlines()
    assert 1 <= len(lines) <= 2 and "calib_loss" in json.loads(lines[0])


def test_head_only_keeps_encoder(tiny, toy_csv, tmp_path):
    base = _weights(tiny)
    ds.finetune(
        toy_csv, Ticket, base=str(tiny), out=str(tmp_path / "h"), train="head", epochs=1, device="cpu", verbose=False
    )
    after = _weights(tmp_path / "h")
    enc = [k for k in base if k.startswith("encoder.")]
    head = [k for k in base if k.startswith(("head.", "scorer."))]
    assert all(torch.equal(base[k].half(), after[k]) for k in enc)
    assert any(not torch.equal(base[k].half(), after[k]) for k in head)
    assert all(torch.equal(base[k].half(), after[k]) for k in base if k.startswith("act_head."))


def test_full_training_changes_encoder(tiny, toy_csv, tmp_path):
    base = _weights(tiny)
    rep = ds.finetune(
        toy_csv,
        Ticket,
        base=str(tiny),
        out=str(tmp_path / "f"),
        train="full",
        epochs=1,
        lr=1e-3,
        device="cpu",
        verbose=False,
        loss="proper",
    )
    after = _weights(tmp_path / "f")
    assert any(not torch.equal(base[k].half(), after[k]) for k in base if k.startswith("encoder."))
    assert rep.details["provenance"]["train"] == "full" and rep.details["provenance"]["loss"] == "proper"


def test_resume_continues(tiny, toy_csv, tmp_path):
    out = str(tmp_path / "r")
    ds.finetune(toy_csv, Ticket, base=str(tiny), out=out, epochs=1, device="cpu", verbose=False)
    msgs = []
    rows = load(toy_csv, Ticket)
    import laya

    tr_rows, ca, _ = split(rows)
    agent = laya.load(str(tiny), device="cpu")
    items, _ = tr.build_items(agent, ca)
    with pytest.raises(ValueError, match="can't resume"):
        tr.train(agent, tr_rows, items, out, tr.Settings(epochs=2, resume=True), torch.device("cpu"), msgs.append)
    s = tr.Settings(epochs=2, resume=True, early_stop=False, head_only=True, accum=1, lr_head=5e-4)
    run = tr.train(agent, tr_rows, items, out, s, torch.device("cpu"), msgs.append)
    assert msgs[0] == "resumed at epoch 1" and [e["epoch"] for e in run["log"]] == [1, 2]
    assert not list((tmp_path / "r" / "checkpoint_latest").glob("*.pt"))


def test_resume_never_unpickles_old_state(tiny, toy_csv, tmp_path, monkeypatch):
    out = tmp_path / "old"
    (out / "checkpoint_latest").mkdir(parents=True)
    (out / "checkpoint_latest" / "state.pt").write_bytes(b"not loaded")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("torch.load must not be called"))
    rows = load(toy_csv, Ticket)
    import laya

    tr_rows, ca, _ = split(rows)
    agent = laya.load(str(tiny), device="cpu")
    items, _ = tr.build_items(agent, ca)
    with pytest.raises(ValueError, match="pickled files are never loaded"):
        tr.train(
            agent, tr_rows, items, str(out), tr.Settings(epochs=1, resume=True, head_only=True), torch.device("cpu")
        )


def test_typed_decisions_without_schema(tiny, tmp_path):
    recs = []
    for i, text in enumerate(corpus(40)):
        t = truth(text)
        recs.append(
            {
                "id": str(i),
                "state": text,
                "questions": {
                    "team": {
                        "type": "choice",
                        "instructions": "Which team?",
                        "criteria": ["billing", "technical", "sales"],
                    },
                    "lvl": {"type": "score", "instructions": "rate", "criteria": ["low", "high"]},
                },
                "gold": {"team": {"label": t["team"]}, "lvl": {"label": "1" if t["wants_refund"] else "0"}},
            }
        )
    p = tmp_path / "td.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs))
    rep = ds.finetune(
        str(p), base=str(tiny), out=str(tmp_path / "td"), epochs=1, device="cpu", verbose=False, max_steps=2
    )
    assert {r["field"] for r in rep.rows} == {"team", "lvl", "all"} and "mae" in rep.rows[1]
    assert "schema" not in rep.details["provenance"]


def test_bad_options(tiny, toy_csv, tmp_path):
    for kw, match in (({"train": "most"}, "train must be"), ({"loss": "mse"}, "loss must be")):
        with pytest.raises(ValueError, match=match):
            ds.finetune(toy_csv, Ticket, base=str(tiny), out=str(tmp_path / "x"), **kw)


def test_skips_items_that_do_not_fit(tiny):
    import laya

    agent = laya.load(str(tiny), device="cpu")
    agent.cfg["max_len"] = 10
    long = {"q": {"type": "choice", "instructions": "?", "criteria": ["a " * 10, "b " * 10, "c " * 10, "d " * 10]}}
    rows = [ds_row("x", long, {"q": [1, 0, 0, 0]})]
    items, skipped = tr.build_items(agent, rows)
    assert items == [] and skipped == [("x", "q")]
    with pytest.raises(ValueError, match="targets for"):
        tr.build_items(agent, [ds_row("y", long, {"q": [1, 0]})])


def ds_row(rid, questions, targets):
    from decisionsmith.training.data import Row

    return Row(rid, "text", questions, targets)


def test_option_shuffle_keeps_targets_aligned(tiny):
    import laya

    agent = laya.load(str(tiny), device="cpu")
    q = {"q": {"type": "choice", "instructions": "?", "criteria": ["billing", "technical", "sales"]}}
    plain, _ = tr.build_items(agent, [ds_row("x", q, {"q": [0.0, 0.0, 1.0]})])
    shuffled, _ = tr.build_items(agent, [ds_row("x", q, {"q": [0.0, 0.0, 1.0]})], random.Random(3))
    sales_id = agent.tok.convert_tokens_to_ids("sales")
    pos = shuffled[0].target.index(1.0)
    assert shuffled[0].ids[shuffled[0].markers[pos] + 1] == sales_id
    assert plain[0].ids[plain[0].markers[2] + 1] == sales_id


def _batch():
    return {
        "marker_mask": torch.tensor([[True, True, False], [True, True, True]]),
        "target": torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]]),
        "qtype": torch.tensor([0, 1]),
    }


def test_losses():
    logits = torch.tensor([[2.0, 0.0, 9.0], [0.0, 1.0, 2.0]], requires_grad=True)
    b = _batch()
    ce = tr.loss_fn("ce", logits, b)
    lp0 = torch.log_softmax(torch.tensor([2.0, 0.0]), -1)
    lp1 = torch.log_softmax(torch.tensor([0.0, 1.0, 2.0]), -1)
    manual = (-lp0[0] - 0.5 * lp1[1] - 0.5 * lp1[2]) / 2
    assert float(ce) == pytest.approx(float(manual), rel=1e-5)
    for name in ("proper", "rlcd"):
        loss = tr.loss_fn(name, logits, b, progress=0.5)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match="loss must be"):
        tr.loss_fn("mse", logits, b)
    with pytest.raises(ValueError, match="loss must be"):
        tr.train(None, [], [], "x", tr.Settings(loss="mse"), torch.device("cpu"))


def test_device_and_autocast(monkeypatch):
    assert tr.device("cpu").type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert tr.device().type == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert tr.device().type == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert tr.device().type == "cpu"
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (7, 5))
    ctx, scale = tr._autocast(torch.device("cuda"))
    assert scale is True
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (8, 0))
    assert tr._autocast(torch.device("cuda"))[1] is False


def test_fit_temperatures_and_go():
    items = [tr.Item("r", "q", [], [0, 1], 0, [1.0, 0.0]) for _ in range(12)]
    zs = [np.array([5.0, 0.0]) if i % 2 else np.array([0.0, 5.0]) for i in range(12)]
    per_type, buckets = fmod._fit_temperatures(zs, items)
    assert per_type[0] == pytest.approx(5.0, abs=0.01) and per_type[1] == 1.0 and buckets == {}
    monkey = fmod.BUCKET_MIN
    fmod.BUCKET_MIN = 5
    try:
        assert "choice:2" in fmod._fit_temperatures(zs, items)[1]
    finally:
        fmod.BUCKET_MIN = monkey
    base = {"all": {"n": 200, "accuracy": 0.8, "ece": 0.05}, "team": {"accuracy": 0.9}}
    tuned = {"all": {"n": 200, "accuracy": 0.85, "ece": 0.05}, "team": {"accuracy": 0.95}, "worst": []}
    assert fmod._go(base, tuned, [], None, None) == (True, [])
    worse = {"all": {"n": 50, "accuracy": 0.7, "ece": 0.2}, "team": {"accuracy": 0.8}, "new": {"accuracy": 1}}
    go, reasons = fmod._go(base, worse, [], None, "boom")
    assert not go and len(reasons) == 5 and "boom" in reasons[-1]


def test_ddp_code_path(tiny, toy_csv, tmp_path, monkeypatch):
    import torch.distributed as dist

    calls = []
    monkeypatch.setattr(dist, "init_process_group", lambda backend: calls.append(("init", backend)))
    monkeypatch.setattr(dist, "barrier", lambda: calls.append("barrier"))
    monkeypatch.setattr(dist, "destroy_process_group", lambda: calls.append("destroy"))
    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", lambda m, **kw: m)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    rep = ds.finetune(toy_csv, Ticket, base=str(tiny), out=str(tmp_path / "d"), epochs=1, verbose=False)
    assert rep.title == "worker rank 1 finished" and calls == [("init", "gloo"), "barrier", "destroy"]
    monkeypatch.setenv("RANK", "0")
    calls.clear()
    rep = ds.finetune(toy_csv, Ticket, base=str(tiny), out=str(tmp_path / "d0"), epochs=1, verbose=False)
    assert rep.go is False and calls == [("init", "gloo"), "barrier", "destroy"]


def test_needs_laya(monkeypatch, toy_csv):
    import builtins

    real = builtins.__import__

    def no_torch(name, *a, **k):
        if name == "laya":
            raise ImportError
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    with pytest.raises(ds.EngineError, match="needs laya"):
        ds.finetune(toy_csv, Ticket)


def test_max_steps_and_cuda_scaler_paths(tiny, toy_csv, tmp_path, monkeypatch):
    import laya

    rows = load(toy_csv, Ticket)
    tr_rows, ca, _ = split(rows)
    agent = laya.load(str(tiny), device="cpu")
    items, _ = tr.build_items(agent, ca)
    run = tr.train(
        agent,
        tr_rows,
        items,
        str(tmp_path / "m"),
        tr.Settings(epochs=3, max_steps=2, accum=1, early_stop=False),
        torch.device("cpu"),
        lambda m: None,
    )
    assert run["steps"] == 2

    class Scaler:
        def __init__(self, *a, **k):
            self.used = []

        def scale(self, loss):
            return loss

        def unscale_(self, opt):
            pass

        def step(self, opt):
            opt.step()

        def update(self):
            pass

    monkeypatch.setattr(tr, "_autocast", lambda dev: (nullcontext(), True))
    monkeypatch.setattr(torch.amp, "GradScaler", Scaler)
    agent2 = laya.load(str(tiny), device="cpu")
    fake_cuda = torch.device("cpu")
    monkeypatch.setattr(tr, "_is_cuda", lambda dev: True)
    agent2.model.encoder.gradient_checkpointing_enable = lambda **kw: None
    run = tr.train(agent2, tr_rows, items, str(tmp_path / "c"), tr.Settings(epochs=1), fake_cuda, lambda m: None)
    assert run["steps"] >= 1 and agent2.model.head_checkpointing is True


def test_buckets_and_failed_reload_block_go(tiny, toy_csv, tmp_path, monkeypatch):
    import os

    import laya

    out = tmp_path / "b"
    real = laya.load

    def load(path, *a, **k):
        if path == os.path.abspath(out):
            raise RuntimeError("broken checkpoint")
        return real(path, *a, **k)

    monkeypatch.setattr(fmod, "BUCKET_MIN", 1)
    monkeypatch.setattr(laya, "load", load)
    rep = ds.finetune(toy_csv, Ticket, base=str(tiny), out=str(out), epochs=1, device="cpu", verbose=False)
    cfg = json.loads((out / "rl_agent_config.json").read_text())
    assert cfg["temperature_by_options"] == rep.details["temperatures"]["buckets"] != {}
    assert rep.go is False and any("does not load in laya: broken checkpoint" in r for r in rep.reasons)


def test_train_with_no_steps_keeps_weights(tiny, toy_csv):
    import laya

    tr_rows, ca, _ = split(load(toy_csv, Ticket))
    agent = laya.load(str(tiny), device="cpu")
    before = {k: v.clone() for k, v in agent.model.state_dict().items()}
    items, _ = tr.build_items(agent, ca)
    run = tr.train(agent, tr_rows, items, "unused", tr.Settings(max_steps=0), torch.device("cpu"), print)
    assert run["steps"] == 0 and run["log"] == []
    assert all(torch.equal(before[k], v) for k, v in agent.model.state_dict().items())


def test_go_counts_only_rows_with_the_field(monkeypatch):
    from decisionsmith.schema import compile_schema
    from decisionsmith.training.data import Row

    monkeypatch.setattr(fmod, "GO_PER_OPTION", 1)
    rows = [Row("a", "t", {}, {}), Row("b", "t", {}, {"team": [1.0, 0.0, 0.0]})]
    ok = {"all": {"n": 100, "ece": 0.0, "accuracy": 1.0}}
    go, reasons = fmod._go({"all": {"accuracy": 0.5}}, ok, rows, compile_schema(Ticket), None)
    assert not go and reasons == [
        "team: options ['technical', 'sales'] have fewer than 1 test rows",
        "wants_refund: options ['false', 'true'] have fewer than 1 test rows",
    ]
