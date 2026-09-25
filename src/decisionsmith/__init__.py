"""decisionsmith: use and fine-tune System One models (Jev, Laya) on your data, with an LLM as the teacher."""

from typing import Any

from . import testing
from .core import Result, harness
from .engines import Engine, EngineError
from .engines import LLMEngine as LLM
from .predictor import Model, load, model
from .report import Report
from .schema import Options, Scale
from .status import Status

__version__ = "0.1.0"


def finetune(data: Any, schema: Any = None, **options: Any) -> Report:
    """Fine-tune Laya on labelled data (CSV, JSONL, or rows) and write a checkpoint that loads in `laya.load()`.

    ds.finetune("tickets.csv", Ticket, base="laya", out="runs/v1")
    """
    from .training.finetuning import finetune as _finetune

    return _finetune(data, schema, **options)


def golden(
    source: Any, teacher: Any, n: int = 500, strategy: str = "uncertain", **options: Any
) -> list[dict[str, Any]]:
    """Pick the texts most worth labelling, have your main LLM label them, and write `golden.csv` to review.

        rows = ds.golden("decisions.db", teacher="claude-opus-5", schema=Ticket, n=500)

    `source` is a harness (or its `.db` log), a CSV/JSONL/.txt of texts, or a list of texts. `strategy`:
    `uncertain` (lowest student confidence first), `disagree` (student vs teacher), `diverse` (across labels and
    lengths) or `random`. A `test` share (0.2) is marked `split=test`; training skips it and `evaluate` uses only it.
    """
    from .golden_set import golden as _golden

    return _golden(source, teacher, n, strategy, **options)


def bench(schema: Any, data: Any, engines: Any, **options: Any) -> Report:
    """Compare engines on labelled data: accuracy, macro-F1, ECE, latency and cost, per engine and field."""
    from .benchmark import bench as _bench

    return _bench(schema, data, engines, **options)


__all__ = [
    "LLM",
    "Engine",
    "EngineError",
    "Model",
    "Options",
    "Report",
    "Result",
    "Scale",
    "Status",
    "__version__",
    "bench",
    "finetune",
    "golden",
    "harness",
    "load",
    "model",
    "testing",
]
