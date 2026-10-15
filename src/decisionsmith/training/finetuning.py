"""`ds.finetune`: labelled data -> a calibrated Laya checkpoint that loads unchanged in `laya.load()`."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Sequence
from typing import Any

from ..engines import need, resolve_laya
from ..files import folder_name, write_jsonl
from ..report import Report, metrics
from ..schema import argmax, compile_schema
from . import calibrate
from . import data as data_mod

HEAD_ONLY_BELOW = 1000
HEAD_LR = 5e-4
BUCKET_MIN = 2000
GO_TEST_DECISIONS, GO_PER_OPTION, GO_ECE, GO_FIELD_DROP = 100, 10, 0.10, 0.02


def _temps(agent: Any, qtype: int, k: int) -> float:
    from laya.common import temp_bucket

    return float(agent.temperature_by_options.get(temp_bucket(qtype, k), agent.temperature[qtype]))


def _fit(sel: list[tuple[Any, list[float]]]) -> float:
    return calibrate.fit_temperature([z for z, _ in sel], [t for _, t in sel])


def _fit_temperatures(zs: Sequence[Any], items: Sequence[Any]) -> tuple[list[float], dict[str, float]]:
    from laya.common import temp_bucket

    per_type = [1.0, 1.0, 1.0]
    by_type: dict[int, list[tuple[Any, list[float]]]] = {}
    by_bucket: dict[str, list[tuple[Any, list[float]]]] = {}
    for z, it in zip(zs, items):
        by_type.setdefault(it.qtype, []).append((z, it.target))
        by_bucket.setdefault(temp_bucket(it.qtype, len(z)), []).append((z, it.target))
    for qt, sel in by_type.items():
        if len(sel) >= 10:
            per_type[qt] = _fit(sel)
    return per_type, {key: _fit(sel) for key, sel in by_bucket.items() if len(sel) >= BUCKET_MIN}


def _evaluate(
    zs: Sequence[Any], items: Sequence[Any], temp: Callable[[int, int], float], rows: dict[str, Any]
) -> dict[str, Any]:
    by_q: dict[str, dict[str, list[Any]]] = {}
    losses = []
    for z, it in zip(zs, items):
        p = calibrate.softmax(z, temp(it.qtype, len(z)))
        d = by_q.setdefault(it.qid, {"pred": [], "gold": [], "qtype": [it.qtype]})
        d["pred"].append(p)
        d["gold"].append(it.target)
        losses.append((-sum(g * math.log(min(1.0, max(1e-12, v))) for g, v in zip(it.target, p)), it, p))
    out: dict[str, Any] = {q: metrics(d["pred"], d["gold"], ordinal=d["qtype"][0] == 1) for q, d in by_q.items()}
    allp = [p for d in by_q.values() for p in d["pred"]]
    allg = [g for d in by_q.values() for g in d["gold"]]
    out["all"] = metrics(allp, allg)
    worst = sorted(losses, key=lambda x: -x[0])[:10]
    out["worst"] = [
        {
            "row": it.row,
            "field": it.qid,
            "loss": loss,
            "text": str(rows[it.row].text)[:200],
            "gold": argmax(it.target),
            "pred": argmax(p),
        }
        for loss, it, p in worst
    ]
    return out


def _base_hashes(base_id: str) -> set[str]:
    """Texts the base checkpoint was already trained on, so a retrained model remembers them too."""
    cfg = os.path.join(base_id, "rl_agent_config.json")
    if not os.path.isfile(cfg):
        return set()
    with open(cfg, encoding="utf-8") as f:
        return set((json.load(f).get("decisionsmith") or {}).get("text_hashes") or [])


def _data_hash(rows: Sequence[data_mod.Row]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps([r.id, r.text, r.targets], sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def _go(
    base: dict[str, Any],
    tuned: dict[str, Any],
    test_rows: Sequence[data_mod.Row],
    schema: Any,
    loads: str | None,
) -> tuple[bool, list[str]]:
    reasons = []
    n = tuned["all"]["n"]
    if n < GO_TEST_DECISIONS:
        reasons.append("test split has %d decisions; want %d (add data)" % (n, GO_TEST_DECISIONS))
    if schema is not None:
        for name, f in schema.fields.items():
            counts = {k: 0 for k in f.labels}
            for r in test_rows:
                if name in r.targets:
                    counts[f.labels[argmax(r.targets[name])]] += 1
            low = [k for k, c in counts.items() if c < GO_PER_OPTION]
            if low:
                reasons.append("%s: options %s have fewer than %d test rows" % (name, low, GO_PER_OPTION))
    if tuned["all"]["ece"] > GO_ECE:
        reasons.append("calibration error %.3f is above %.2f" % (tuned["all"]["ece"], GO_ECE))
    if tuned["all"]["accuracy"] <= base["all"]["accuracy"]:
        reasons.append(
            "fine-tuned accuracy %.3f does not beat the base %.3f" % (tuned["all"]["accuracy"], base["all"]["accuracy"])
        )
    for q in tuned:
        if q in ("all", "worst") or q not in base:
            continue
        if tuned[q]["accuracy"] < base[q]["accuracy"] - GO_FIELD_DROP:
            reasons.append("%s got worse: %.3f vs base %.3f" % (q, tuned[q]["accuracy"], base[q]["accuracy"]))
    if loads:
        reasons.append("the saved checkpoint does not load in laya: %s" % loads)
    return not reasons, reasons


def _model_card(out: str, base_id: str, sub: str | None, settings: dict[str, Any], tuned: dict[str, Any]) -> str:
    base_name = base_id + ("/" + sub if sub else "")
    acc = tuned["all"].get("accuracy")
    return (
        "---\nlicense: apache-2.0\nbase_model: %s\nlibrary_name: laya\ntags: [laya, system-one, decisionsmith]\n---\n\n"
        "# %s\n\nA Laya checkpoint fine-tuned with decisionsmith from `%s`.\n\n"
        "- test accuracy: %s (held-out split; see report.json)\n- training: %s\n\n"
        'Load it with `laya.load("%s")` or `ds.harness(..., student="laya:%s")`.\n\n'
        "Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai "
        "Innovations. Not affiliated with TypeSafe AI or Convai Innovations.\n"
    ) % (
        base_name,
        os.path.basename(os.path.abspath(out)),
        base_name,
        "%.3f" % acc if acc is not None else "-",
        json.dumps(settings),
        out,
        out,
    )


def finetune(
    data: Any,
    schema: Any = None,
    *,
    base: str = "laya",
    out: str = "runs/v1",
    train: str = "auto",
    epochs: int = 4,
    batch: int = 8,
    accum: int | None = None,
    lr: float | None = None,
    loss: str = "ce",
    seed: int = 0,
    device: str | None = None,
    group_by: str | None = None,
    resume: bool = False,
    max_steps: int | None = None,
    verbose: bool = True,
) -> Report:
    """Fine-tune Laya on labelled data and write a checkpoint that loads unchanged in `laya.load()`.

    `train="auto"` trains only the decision head below 1,000 rows (fast, fine on CPU/MPS) and the whole model
    above. Temperatures are fitted on a held-out split; the report compares base vs fine-tuned on a test split.
    """
    t_start = time.time()
    laya, torch = need("laya", "fine-tuning needs laya and torch", "laya", "laya", "torch")
    import torch.distributed as dist

    from . import train as tr

    if train not in ("auto", "head", "full"):
        raise ValueError("train must be 'auto', 'head' or 'full', got %r" % train)
    if loss not in tr.LOSSES:
        raise ValueError("loss must be one of %s, got %r" % (tr.LOSSES, loss))
    compiled = compile_schema(schema) if schema is not None else None
    rows = data_mod.load(data, compiled, group_by=group_by)
    train_rows, calib_rows, test_rows = data_mod.split(rows, seed=seed)
    head_only = train == "head" or (train == "auto" and len(train_rows) < HEAD_ONLY_BELOW)

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    say: Callable[[str], None] = print if verbose and rank == 0 else (lambda _m: None)
    if world > 1:
        local = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        dev = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    else:
        dev = tr.device(device)

    base_id, sub = resolve_laya(base)
    say("loading %s on %s" % (base, dev))
    agent = laya.load(base_id, device=str(dev), subfolder=sub)
    pad = agent.tok.pad_token_id
    calib_items, skipped_c = tr.build_items(agent, calib_rows)
    test_items, skipped_t = tr.build_items(agent, test_rows)
    by_id = {r.id: r for r in rows}

    base_z = tr.predict_logits(agent.model, test_items, pad, dev)
    base_eval = _evaluate(base_z, test_items, lambda qt, k: _temps(agent, qt, k), by_id)
    base_acc = base_eval["all"].get("accuracy", float("nan"))
    say("base accuracy on test: %.3f  (%d decisions)" % (base_acc, len(test_items)))
    settings = tr.Settings(
        epochs=epochs,
        batch=batch,
        accum=accum or (1 if head_only else 4),
        loss=loss,
        head_only=head_only,
        lr_encoder=lr or 2.5e-5,
        lr_head=(lr or HEAD_LR) if head_only else 1e-4,
        seed=seed,
        max_steps=max_steps,
        resume=resume,
    )
    what = "the head" if head_only else "the full model"
    say("training %s on %d rows (%s, loss=%s)" % (what, len(train_rows), "resume" if resume else "fresh", loss))
    run = tr.train(agent, train_rows, calib_items, out, settings, dev, say, rank=rank, world=world)
    if world > 1:
        dist.barrier()
        if rank != 0:
            dist.destroy_process_group()
            return Report("finetune", "worker rank %d finished" % rank, [])

    with torch.no_grad():
        for p in agent.model.parameters():
            p.copy_(p.half().float())
    calib_z = tr.predict_logits(agent.model, calib_items, pad, dev)
    per_type, buckets = _fit_temperatures(calib_z, calib_items)
    from laya.common import temp_bucket

    def tuned_temp(qt: int, k: int) -> float:
        return buckets.get(temp_bucket(qt, k), per_type[qt])

    t0 = time.perf_counter()
    test_z = tr.predict_logits(agent.model, test_items, pad, dev)
    ms_per_decision = (time.perf_counter() - t0) * 1000 / max(1, len(test_items))
    tuned_eval = _evaluate(test_z, test_items, tuned_temp, by_id)

    os.makedirs(out, exist_ok=True)
    from safetensors.torch import save_file

    save_file(
        {k: v.half().contiguous().cpu() for k, v in agent.model.state_dict().items()},
        os.path.join(out, "model.safetensors"),
    )
    agent.model.encoder.config.save_pretrained(os.path.join(out, "encoder"))
    agent.tok.save_pretrained(os.path.join(out, "tokenizer"))
    cfg = dict(agent.cfg)
    cfg.pop("temperature_by_options", None)
    cfg.update({"temperature": per_type, "fine_tuned": True, "model_name": os.path.basename(os.path.abspath(out))})
    if buckets:
        cfg["temperature_by_options"] = buckets
    from .. import __version__

    provenance = {
        "version": __version__,
        "laya_version": getattr(laya, "__version__", None),
        "base": folder_name(base) if os.path.isdir(os.path.expanduser(base)) else base,
        "base_id": folder_name(base_id) if os.path.isdir(base_id) else base_id,
        "subfolder": sub,
        "data_hash": _data_hash(rows),
        "text_hashes": sorted({data_mod.text_hash(r.text) for r in rows} | _base_hashes(base_id)),
        "seed": seed,
        "train": "head" if head_only else "full",
        "loss": loss,
        "epochs_run": len(run["log"]),
        "rows": {"train": len(train_rows), "calib": len(calib_rows), "test": len(test_rows)},
    }
    if compiled is not None:
        provenance["schema"] = {"name": compiled.name, "questions": compiled.questions()}
    cfg["decisionsmith"] = provenance
    with open(os.path.join(out, "rl_agent_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    write_jsonl(os.path.join(out, "train_log.jsonl"), run["log"])

    load_error = None
    agent.model = None
    gc.collect()
    try:
        laya.load(os.path.abspath(out), device="cpu")
    except Exception as e:
        load_error = str(e)[:300]

    go, reasons = _go(base_eval, tuned_eval, test_rows, compiled, load_error)
    table = []
    for q in [k for k in tuned_eval if k not in ("worst", "all")] + ["all"]:
        b, t = base_eval.get(q, {}), tuned_eval[q]
        row = {
            "field": q,
            "test": t.get("n"),
            "base_acc": b.get("accuracy"),
            "accuracy": t.get("accuracy"),
            "macro_f1": t.get("macro_f1"),
            "base_ece": b.get("ece"),
            "ece": t.get("ece"),
        }
        if "mae" in t:
            row["mae"] = t["mae"]
        table.append(row)
    details = {
        "base": base_eval,
        "finetuned": tuned_eval,
        "temperatures": {"per_type": per_type, "buckets": buckets},
        "skipped": [list(x) for x in skipped_c + skipped_t],
        "train_ids": [r.id for r in train_rows],
        "train_log": run["log"],
        "ms_per_decision": ms_per_decision,
        "seconds": round(time.time() - t_start, 1),
        "provenance": provenance,
    }
    title = "finetune: %s -> %s" % (base, out)
    report = Report("finetune", title, table, go=go, reasons=reasons, path=os.path.abspath(out), details=details)
    report.save(os.path.join(out, "report.json"))
    report.save(os.path.join(out, "report.html"))
    with open(os.path.join(out, "MODEL_CARD.md"), "w", encoding="utf-8") as f:
        f.write(_model_card(out, base_id, sub, {"train": provenance["train"], "loss": loss}, tuned_eval))
    if world > 1:
        dist.destroy_process_group()
    return report
