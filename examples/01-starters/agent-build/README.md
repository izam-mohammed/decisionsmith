# Build a model with a coding agent

A coding agent labels the golden set itself through the MCP tools (no LLM API key), with a second, independent answer on a share of the rows, then trains, evaluates and saves; here a scripted stand-in plays the agent.

```python
"""The minimal agent flow: ds.golden(teacher="agent") -> the agent labels through the MCP tools -> train -> evaluate."""

import csv
from pathlib import Path

import decisionsmith as ds
from decisionsmith import mcp

TOY = Path(__file__).parents[2] / "data" / "toy.csv"
rows = list(csv.DictReader(TOY.open(encoding="utf-8")))
labels = ["billing", "technical", "sales"]

# pick the rows worth labelling into a session file instead of calling an LLM
ds.golden([r["text"] for r in rows], teacher="agent:claude-code", schema=labels, strategy="diverse", n=120)

# a scripted stand-in for the coding agent: it answers from the toy file's labels, where a real agent reads the text
known = {r["text"]: r["team"] for r in rows}
session = "golden.session.json"
while (batch := mcp.golden_batch(session))["items"]:  # a share of texts comes back under new ids for a second answer
    mcp.golden_submit(session, [{"id": i["id"], "answers": {"label": known[i["text"]]}} for i in batch["items"]])
print(mcp.golden_finish(session)["message"])  # writes golden.csv

model = ds.model(labels)
model.train("golden.csv")  # split=test rows are held out
print(model.evaluate("golden.csv"))  # ends with the agreement with the agent's labels
```

| file | what it shows |
|---|---|
| [`cli.py`](cli.py) | The same flow from the command line: `decisionsmith golden --teacher agent`, the agent labels, `--finish`, train, |
| [`main.py`](main.py) | The minimal agent flow: ds.golden(teacher="agent") -> the agent labels through the MCP tools -> train -> evaluate. |
| [`realistic.py`](realistic.py) | Two fields, every MCP tool: data_check, labelling with skips, synthetic rows with a second answer, finetune, |

## Run

```bash
uv add "decisionsmith[laya,mcp]"
uv run python examples/01-starters/agent-build/cli.py
uv run python examples/01-starters/agent-build/main.py
uv run python examples/01-starters/agent-build/realistic.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
