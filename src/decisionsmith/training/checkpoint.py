"""Resume state without pickle: tensors in `state.safetensors`, everything else in `state.json`.

A pickled `torch.load` file can run code when it is loaded, so a resume folder copied from elsewhere must never be
unpickled. Only tensors, dicts, lists, tuples, numbers, strings, booleans and None are stored.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import torch

TENSORS, META, LEGACY = "state.safetensors", "state.json", "state.pt"


def _pack(obj: Any, key: str, tensors: dict[str, torch.Tensor]) -> Any:
    if isinstance(obj, torch.Tensor):
        tensors[key] = obj.detach().cpu().contiguous()
        return {"tensor": key}
    if isinstance(obj, dict):
        return {
            "dict": [
                [_pack(k, "%s.k%d" % (key, i), tensors), _pack(v, "%s.%d" % (key, i), tensors)]
                for i, (k, v) in enumerate(obj.items())
            ]
        }
    if isinstance(obj, (list, tuple)):
        return {
            "list": [_pack(v, "%s.%d" % (key, i), tensors) for i, v in enumerate(obj)],
            "tuple": isinstance(obj, tuple),
        }
    if isinstance(obj, float) and not math.isfinite(obj):
        return {"float": str(obj)}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError("can't store a %s in the resume state" % type(obj).__name__)


def _unpack(obj: Any, tensors: dict[str, torch.Tensor]) -> Any:
    if not isinstance(obj, dict):
        return obj
    if "tensor" in obj:
        return tensors[obj["tensor"]]
    if "float" in obj:
        return float(obj["float"])
    if "dict" in obj:
        return {_unpack(k, tensors): _unpack(v, tensors) for k, v in obj["dict"]}
    items = [_unpack(v, tensors) for v in obj["list"]]
    return tuple(items) if obj["tuple"] else items


def save(folder: str, state: dict[str, Any]) -> None:
    from safetensors.torch import save_file

    os.makedirs(folder, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    meta = _pack(state, "s", tensors)
    save_file(tensors, os.path.join(folder, TENSORS))
    with open(os.path.join(folder, META), "w", encoding="utf-8") as f:
        json.dump(meta, f)


def exists(folder: str) -> bool:
    return os.path.exists(os.path.join(folder, META)) or os.path.exists(os.path.join(folder, LEGACY))


def load(folder: str) -> dict[str, Any]:
    from safetensors.torch import load_file

    if not os.path.exists(os.path.join(folder, META)):
        raise ValueError(
            "can't resume from %s: it holds state.pt from an older decisionsmith, and pickled files are never "
            "loaded because they can run code; delete that folder or train to a new --out" % folder
        )
    with open(os.path.join(folder, META), encoding="utf-8") as f:
        meta = json.load(f)
    return dict(_unpack(meta, load_file(os.path.join(folder, TENSORS))))
