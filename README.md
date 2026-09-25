# decisionsmith

**Use and fine-tune System One models (Jev, Laya) on your data.**
Start with an LLM. End with a fast decision model you trained. One line in between.

```bash
uv add "decisionsmith[all]"
```

```python
import decisionsmith as ds

h = ds.harness(Ticket, teacher="claude-sonnet-5", student="laya")
h("You charged me twice, refund now!")
# Ticket(team='billing', wants_refund=True)
```

That's it. `Ticket` is a normal Pydantic model:

```python
from typing import Annotated, Literal
from pydantic import BaseModel, Field


class Ticket(BaseModel):
    team: Annotated[
        Literal["billing", "technical", "sales"],
        ds.Options(billing="payments, refunds", technical="bugs, outages", sales="pricing"),
    ]
    wants_refund: bool = Field(description="Does the customer ask for their money back?")
```

`Literal` / `Enum` → pick one · `bool` → yes/no · `Annotated[Literal["low", "medium", "high"], ds.Scale]` → a rating.

## Why

Decision models like Laya are fast and free to run, and weak zero-shot (0.36 accuracy on typed-decisions before
fine-tuning, 0.77 after, per Laya's benchmarks). LLMs are good zero-shot, and slow and paid per call.
decisionsmith starts with the LLM, logs every answer, fine-tunes the fast model on those answers, and moves traffic
to it field by field once the numbers say it's ready.

## The loop

```python
h = ds.harness(Ticket, teacher="claude-sonnet-5", student="laya", mode="shadow")  # teacher answers, student measured
h.many(texts)  # every decision is logged (decisions.db)
print(h.status())  # per field: agreement, sure rate, accuracy when sure, what to do next
h.adapt()  # calibrate the student's confidence + pick safe thresholds (seconds, any engine)
h.finetune()  # train Laya on the logged answers; switches only if it beats the current student
```

```
team:         shadow · student agrees 94% · sure on 71% · accuracy when sure 97% (vs teacher) -> ready for cascade
wants_refund: shadow · student agrees 81% · 420 labelled -> ready to finetune: run h.finetune()
```

Then move a field when it's ready: `mode={"team": "cascade", "wants_refund": "shadow"}`.

## Four modes, per field

| mode | who answers | use when |
|---|---|---|
| `teacher` | the LLM | day 0 |
| `shadow` | the LLM; the student runs silently and is measured | measuring the student |
| `cascade` (default) | the student when it's sure, else the LLM (5% of sure answers are audited) | student proven on some fields |
| `student` | the student | student proven everywhere |

If an engine fails, the other one answers and the result says `sure=False`. Nothing crashes in the middle.

## Connect anything

| engine | string | notes |
|---|---|---|
| Any LLM | `"claude-sonnet-5"`, `"gpt-5"`, `"ollama/qwen3"` … | via LiteLLM |
| Jev (TypeSafe) | `"jev"` | hosted, needs `TYPESAFE_API_KEY`; can be adapted, not fine-tuned |
| Laya (local) | `"laya"`, `"laya:multilingual"`, `"laya:./runs/v1"` | runs on your machine |
| Laya server / laya.cpp | `"systemone:http://localhost:8000"` | any Jev-compatible `/v1/systemone` endpoint |

Any engine can be the teacher or the student.

## Fine-tune with labels you already have

```bash
decisionsmith finetune tickets.csv --schema app.py:Ticket --out runs/v1
```

A CSV needs a `text` column and one column per field. Small data (under 1,000 rows) trains only the decision
head, fast even on a laptop; bigger data trains the whole model (use a GPU, or the Kaggle/Colab notebook in
`examples/`). You get `runs/v1/`: a plain Laya checkpoint (`laya.load("runs/v1")` works), a report comparing it
with the base model on held-out data, and a go/no-go verdict with the reasons.

## Compare engines on your data

```bash
decisionsmith bench tickets.csv --schema app.py:Ticket --engines claude-sonnet-5,jev,laya,laya:./runs/v1
```

## From agents

`uvx --from "decisionsmith[all]" decisionsmith mcp` gives Claude Code, Codex, Cursor and other MCP clients the
harness, bench, status, finetune and label as tools. There's also a Claude Code plugin; see
[docs/agents.md](docs/agents.md).

## Docs

[Quickstart](docs/quickstart.md) · [Fine-tuning](docs/finetune.md) · [Guide: engines, modes, status, adapt,
bench](docs/guide.md) · [FAQ](docs/faq.md) · [Integrations](docs/integrations.md)

Runnable [examples](examples/) (gallery, every one runs offline) and [notebooks](notebooks/).

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations.
Jev and TypeSafe are trademarks of their owners; decisionsmith is an independent project, not affiliated with
TypeSafe AI or Convai Innovations.
