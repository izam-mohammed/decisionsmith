# Quickstart

> **TL;DR**: define a Pydantic model, give the harness an LLM as teacher and Laya as student, run in `shadow`
> mode, read `status()`, then `adapt()` and `finetune()`. Everything runs locally except the LLM calls.

```bash
pip install "decisionsmith[all]"
export ANTHROPIC_API_KEY=...          # or OPENAI_API_KEY, GEMINI_API_KEY, ...; or a local Ollama model
```

```python
from typing import Annotated, Literal
from pydantic import BaseModel, Field
import decisionsmith as ds


class Ticket(BaseModel):
    """A customer support ticket."""

    team: Annotated[
        Literal["billing", "technical", "sales"],
        ds.Options(billing="payments, refunds", technical="bugs, outages", sales="pricing"),
    ]
    wants_refund: bool = Field(description="Does the customer ask for their money back?")


h = ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya", mode="shadow")
h("You charged me twice, refund now!")  # the teacher answers; Laya runs silently and is logged
```

| step | call | what happens |
|---|---|---|
| 1 | `h.many(texts)` | decisions logged to `decisions.db` (text, both engines' probabilities, source) |
| 2 | `print(h.status())` | per field: agreement, sure rate, accuracy when sure, advice |
| 3 | `h.label(r.id, team="sales")` | a human correction; beats the teacher everywhere |
| 4 | `h.adapt()` | calibrates Laya's confidence and picks per-field thresholds (seconds) |
| 5 | `h.finetune()` | trains Laya on the log; the harness switches only if the new model is better |
| 6 | `mode={"team": "cascade"}` | move a field when `status()` says it's ready |

The first `laya` call downloads the checkpoint (about 1.7 GB) into the Hugging Face cache.
Try it without keys: `DS_OFFLINE=1 python examples/01-starters/quickstart/main.py` (LLM answers come from a local
stand-in; see [teachers.md](teachers.md#offline-mode)).

Next: [fine-tuning](finetune.md) · [guide](guide.md) · [FAQ](faq.md)
