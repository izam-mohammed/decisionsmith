---
name: decisionsmith
description: "Add fast, calibrated decisions (classify, route, screen, score, yes/no) to code with the decisionsmith harness: an LLM or Jev as teacher, Laya as a fast student, one Pydantic schema. Use when the answer is one of a known set of options. Not for generating text."
---
# decisionsmith

## Default pattern
```python
from typing import Annotated, Literal
from pydantic import BaseModel, Field
import decisionsmith as ds


class Ticket(BaseModel):
    team: Annotated[Literal["billing", "technical", "sales"], ds.Options(billing="payments, refunds")]
    wants_refund: bool = Field(description="Does the customer ask for their money back?")


h = ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya", mode="shadow")
ticket = h(text)  # typed value; h.decide(text) gives source/confidence/id too
```
Field types: `Literal`/`Enum` (pick one), `bool` (yes/no), `Annotated[Literal["low","medium","high"], ds.Scale]` (rating). Free text,
lists and nested models can't be decided; give them defaults and they are skipped.

## Rules
- Decision models fail zero-shot: start new fields in `mode="shadow"` (the teacher answers) until `status()` says ready.
- Ask the user before using a paid teacher, sending data to a hosted engine (Jev, LLM APIs), or starting a long training run.
- Show `status()`, `bench` or finetune report numbers before claiming the student is good; include where it loses.
- Never change a field's mode without the user's approval, and only when `status()` advice says it is ready.
- Jev cannot be fine-tuned; to improve a Jev student use `h.adapt()`; to improve Laya use `h.adapt()` then `h.finetune()`.
- Built on Laya (Apache-2.0, Nandakishor M / Convai Innovations). Not affiliated with TypeSafe AI or Convai.
