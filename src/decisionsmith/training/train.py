"""Training loop on Laya's own sequence builder and model: items, losses, devices, early stop, resume, DDP."""

from __future__ import annotations

import math
import os
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from . import checkpoint
from .calibrate import nll
from .data import Row
from .laya_compat import internal_question

LOSSES = ("ce", "proper", "rlcd")


@dataclass
class Item:
    row: str
    qid: str
    ids: list[int]
    markers: list[int]
    qtype: int
    target: list[float]


@dataclass
class Settings:
    epochs: int = 4
    batch: int = 8
    accum: int = 4
    lr_encoder: float = 2.5e-5
    lr_head: float = 1e-4
    weight_decay: float = 0.01
    loss: str = "ce"
    head_only: bool = False
    seed: int = 0
    max_steps: int | None = None
    early_stop: bool = True
    resume: bool = False
    log: list[dict[str, Any]] = field(default_factory=list)


def _is_cuda(dev: torch.device) -> bool:
    return dev.type == "cuda"


def device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_items(
    agent: Any, rows: Sequence[Row], rng: random.Random | None = None
) -> tuple[list[Item], list[tuple[str, str]]]:
    """Rows -> Laya items at the checkpoint's own max_len/head_max_len; options shuffled when `rng` is given."""
    from laya.common import QTYPES, build_sequence, render_options, serialize_state

    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    tok = agent.tok
    items: list[Item] = []
    skipped: list[tuple[str, str]] = []
    for r in rows:
        state_ids = tok(serialize_state(r.text).replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"]
        for qid, q in r.questions.items():
            internal = internal_question(qid, q)
            k = len(render_options(internal))
            target = r.targets[qid]
            if len(target) != k:
                raise ValueError("row %s question %r: %d targets for %d options" % (r.id, qid, len(target), k))
            order = list(range(k))
            if rng is not None:
                rng.shuffle(order)
            ids, markers = build_sequence(
                tok,
                r.text,
                internal,
                max_len,
                head_max_len,
                option_order=order,
                truncate_left=isinstance(r.text, list),
                state_ids=state_ids,
            )
            if len(markers) != k:
                skipped.append((r.id, qid))
                continue
            items.append(Item(r.id, qid, ids, markers, QTYPES[internal["t"]], [target[i] for i in order]))
    return items, skipped


def _batch(items: Sequence[Item], pad_id: int, dev: torch.device) -> dict[str, torch.Tensor]:
    from laya.common import collate_items

    b = collate_items(
        [[{"ids": it.ids, "markers": it.markers, "qtype": it.qtype, "target": it.target} for it in items]],
        pad_id,
    )
    return {k: v.to(dev) for k, v in b.items() if isinstance(v, torch.Tensor)}


def _autocast(dev: torch.device) -> tuple[Any, bool]:
    if dev.type == "cuda":
        bf16 = torch.cuda.get_device_capability(dev)[0] >= 8
        return torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float16), not bf16
    return torch.autocast("cpu", enabled=False), False


def loss_fn(name: str, logits: torch.Tensor, b: dict[str, torch.Tensor], progress: float = 0.0) -> torch.Tensor:
    from laya.common import proper_reward

    mask = b["marker_mask"]
    target = b["target"]
    logp = torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
    ce = -(target * logp).sum(-1).mean()
    if name == "ce":
        return ce
    if name == "proper":
        return -proper_reward(logp.exp(), target, b["qtype"], mask.float(), w_sph=0.75, w_rps=1.0).mean()
    if name == "rlcd":
        sigma = 0.4 + (0.1 - 0.4) * progress
        k = mask.sum(-1, keepdim=True).float()
        eps = torch.randn((4, *logits.shape), device=logits.device) * sigma * mask
        eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
        z = logits.detach().unsqueeze(0) + eps
        q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
        with torch.no_grad():
            r = proper_reward(q, target.unsqueeze(0), b["qtype"], mask.float(), w_sph=0.75, w_rps=1.0)
            adv = (r - r.mean(0, keepdim=True)) / (r - r.mean(0, keepdim=True)).std().add(1e-6)
        pg = -(adv * (-(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2))).mean()
        return pg + ce
    raise ValueError("loss must be one of %s, got %r" % (LOSSES, name))


@torch.no_grad()
def predict_logits(
    model: Any, items: Sequence[Item], pad_id: int, dev: torch.device, batch: int = 16
) -> list[list[float]]:
    model.eval()
    out: list[list[float]] = []
    ctx, _ = _autocast(dev)
    for i in range(0, len(items), batch):
        chunk = items[i : i + batch]
        b = _batch(chunk, pad_id, dev)
        with ctx:
            logits, _ = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        z = logits.float().cpu().tolist()
        out.extend(z[j][: len(it.markers)] for j, it in enumerate(chunk))
    return out


def _state_dir(out: str) -> str:
    return os.path.join(out, "checkpoint_latest")


def _trainable(model: Any) -> dict[str, torch.nn.Parameter]:
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def train(
    agent: Any,
    train_rows: Sequence[Row],
    calib_items: Sequence[Item],
    out: str,
    s: Settings,
    dev: torch.device,
    say: Callable[[str], None] = print,
    rank: int = 0,
    world: int = 1,
) -> dict[str, Any]:
    """Fine-tune `agent.model` in place. Keeps the weights with the lowest calibration-split loss."""
    if s.loss not in LOSSES:
        raise ValueError("loss must be one of %s, got %r" % (LOSSES, s.loss))
    torch.manual_seed(s.seed)
    model = agent.model
    pad_id = agent.tok.pad_token_id
    for n, p in model.named_parameters():
        p.requires_grad_(not (s.head_only and n.startswith("encoder.")) and not n.startswith("act_head."))
    if not s.head_only and _is_cuda(dev):
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    params = _trainable(model)
    groups = [
        {"params": [p for n, p in params.items() if n.startswith("encoder.")], "lr": s.lr_encoder},
        {"params": [p for n, p in params.items() if not n.startswith("encoder.")], "lr": s.lr_head},
    ]
    optimizer = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=s.weight_decay)
    net: Any = model
    if world > 1:
        net = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dev.index] if dev.type == "cuda" else None, find_unused_parameters=True
        )
    n_items = sum(len(r.questions) for r in train_rows)
    per_epoch = max(1, math.ceil(n_items / world / (s.batch * s.accum)))
    total = per_epoch * s.epochs if s.max_steps is None else min(s.max_steps, per_epoch * s.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total), eta_min=1e-6)
    ctx, use_scaler = _autocast(dev)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if _is_cuda(dev) else None

    start_epoch, step, best_loss = 0, 0, math.inf
    best: dict[str, torch.Tensor] | None = None
    if s.resume and checkpoint.exists(_state_dir(out)):
        st = checkpoint.load(_state_dir(out))
        saved = (st.get("head_only"), st.get("loss"))
        if saved != (s.head_only, s.loss):
            raise ValueError(
                "can't resume: %s was trained with head_only=%s, loss=%s; use the same settings or a new --out"
                % (out, *saved)
            )
        model.load_state_dict(st["model"], strict=False)
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        start_epoch, step, best_loss, best = st["epoch"], st["step"], st["best_loss"], st["best"]
        torch.set_rng_state(st["rng"])
        s.log.extend(st["log"])
        say("resumed at epoch %d" % start_epoch)

    stale = 0
    for epoch in range(start_epoch, s.epochs):
        if s.max_steps is not None and step >= s.max_steps:
            break
        model.train()
        if s.head_only:
            model.encoder.eval()
        items, _ = build_items(agent, train_rows, random.Random(s.seed * 1000 + epoch))
        random.Random(s.seed * 7919 + epoch).shuffle(items)
        items = items[: len(items) // world * world][rank::world]
        t0, total_loss, seen = time.time(), 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for i in range(0, len(items), s.batch):
            b = _batch(items[i : i + s.batch], pad_id, dev)
            with ctx:
                logits, _ = net(
                    b["input_ids"],
                    b["attention_mask"],
                    b["marker_pos"],
                    b["marker_mask"],
                    b["qtype"],
                    detach_encoder=s.head_only,
                )
            loss = loss_fn(s.loss, logits.float(), b, epoch / max(1, s.epochs - 1)) / s.accum
            (scaler.scale(loss) if scaler else loss).backward()
            total_loss += float(loss.detach()) * s.accum
            seen += 1
            if seen % s.accum == 0 or i + s.batch >= len(items):
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if s.max_steps is not None and step >= s.max_steps:
                    break
        calib_loss = _calib_loss(model, calib_items, pad_id, dev)
        entry = {
            "epoch": epoch + 1,
            "step": step,
            "train_loss": total_loss / max(1, seen),
            "calib_loss": calib_loss,
            "seconds": round(time.time() - t0, 1),
        }
        s.log.append(entry)
        say(
            "epoch %d/%d  train loss %.4f  calib loss %.4f  (%.0fs)"
            % (epoch + 1, s.epochs, entry["train_loss"], calib_loss, entry["seconds"])
        )
        if calib_loss < best_loss - 1e-6:
            best_loss, stale = calib_loss, 0
            best = {n: p.detach().to("cpu", copy=True).half() for n, p in params.items()}
        else:
            stale += 1
        if rank == 0:
            checkpoint.save(
                _state_dir(out),
                {
                    "model": {n: p.detach().cpu() for n, p in params.items()},
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "step": step,
                    "best_loss": best_loss,
                    "best": best,
                    "log": s.log,
                    "head_only": s.head_only,
                    "loss": s.loss,
                    "rng": torch.get_rng_state(),
                },
            )
        if s.early_stop and stale >= 1:
            say("calibration loss stopped improving; keeping the best epoch")
            break
    if best is not None:
        model.load_state_dict({n: v.float() for n, v in best.items()}, strict=False)
    for p in model.parameters():
        p.requires_grad_(False)
    return {"steps": step, "best_calib_loss": best_loss, "log": s.log}


def _calib_loss(model: Any, items: Sequence[Item], pad_id: int, dev: torch.device) -> float:
    zs = predict_logits(model, items, pad_id, dev)
    return nll(zs, [it.target for it in items], 1.0) if items else 0.0
