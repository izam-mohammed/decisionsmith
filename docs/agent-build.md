# Build a model with a coding agent

A coding agent (Claude Code, Codex, Gemini CLI, Cursor) labels your golden dataset itself, trains Laya and
evaluates it through decisionsmith's MCP tools, so you need no LLM API key.

**Flow:** the developer flow (define, data, golden, train, evaluate, save), run by the agent. You approve the labels,
the report and the code it writes.

```bash
claude "/decisionsmith:build tickets.csv --labels billing,technical,sales"
```

The `/decisionsmith:build` command comes with the Claude Code plugin (`/plugin marketplace add
izam-mohammed/decisionsmith`, then `/plugin install decisionsmith@decisionsmith`). For other agents, point them at the MCP server
(`uvx decisionsmith mcp`) and the "Build a model" playbook in [AGENTS.md](../AGENTS.md). To run it from a script or
CI: `claude -p "/decisionsmith:build tickets.csv --labels billing,technical,sales"`.

## What the agent does

1. Reads your data and real samples, runs `data_check`, and proposes labels (or a schema). **You approve.**
2. Picks the rows worth labelling into a session: `golden_start`, or `decisionsmith golden texts.csv --labels ...
   --teacher agent`. A share (20%) is held out as `split=test` by a hash of each text.
3. Labels them in batches (`golden_batch`, `golden_submit`). A share of the rows (20%, at least one) comes back in a
   later batch under a new, random id for a second, independent answer; rows where the two answers disagree are
   flagged for you and their labels left blank. In Claude Code a fresh `labeler` subagent takes each batch, and it
   has only the two labelling tools (no files), so it can't see the first answer. Each id takes one answer: a
   repeat or a retry is rejected as "already answered".
4. If an option is short of rows, it writes examples (`golden_add`), using training rows as a guide. They are marked
   synthetic, never used as test rows, and kept only when their second answer agrees. A written text that shares
   80% or more of its words with a test text (Jaccard over lowercased words) is rejected, so the test split stays
   unseen.
5. Writes `golden.csv` (`golden_finish`), trains (`finetune`), evaluates and saves (`evaluate` with `save=`).
   **You see the report and the go/no-go.**
6. Writes the production code (`ds.load` plus `ds.harness(..., collect=...)`) into your app. **You approve.**

The same steps in Python, with a scripted stand-in for the agent so this page runs offline:

```python
import decisionsmith as ds
from decisionsmith import mcp

labels = ["billing", "technical", "sales"]
ds.golden("texts.txt", teacher="agent", schema=labels, strategy="diverse", n=40)  # writes golden.session.json


def agent_label(text):  # the agent reads the text and picks an option; this stand-in uses keywords
    words = text.lower()
    return "billing" if "charge" in words or "refund" in words else "sales" if "price" in words else "technical"


session = "golden.session.json"
batch = mcp.golden_batch(session)  # a share of texts comes back under new ids for a second answer
while batch["items"]:
    answers = [{"id": item["id"], "answers": {"label": agent_label(item["text"])}} for item in batch["items"]]
    mcp.golden_submit(session, answers, agent="claude-code")
    batch = mcp.golden_batch(session)
print(mcp.golden_finish(session)["message"])  # writes golden.csv

model = ds.model(labels)
model.train("golden.csv")  # split=test rows are held out
print(model.evaluate("golden.csv"))  # ends with the agreement with the agent's labels
```

## The MCP tools

| tool | does |
|---|---|
| `golden_start(source, labels or schema, n, strategy)` | picks rows from a log (`.db`) or a file of texts into `golden.session.json` |
| `golden_batch(session, size=20)` | the next texts (second answers mixed in under new ids), the options with their descriptions, and instructions |
| `golden_submit(session, answers, agent)` | checks each answer against the schema; bad ones come back with the reason. `{"id": ..., "skip": "why"}` skips a text (the reason is required). Each id takes one answer |
| `golden_add(session, examples, agent)` | examples the agent wrote, marked synthetic; near copies of test texts are rejected |
| `golden_status(session)` | progress, balance per option, agreement between the two answers, disagreements, skipped rows |
| `golden_finish(session)` | writes `golden.csv` (also `decisionsmith golden --finish golden.session.json`) and closes the session |
| `data_check(path)` | counts per option, repeated texts, texts in both train and test, near copies (a training text sharing 80% or more of its words with a test text), lengths, advice |
| `finetune(data, schema)` | trains Laya; `schema` can be the session file |
| `evaluate(model, data, schema, save=...)` | per field numbers, then by who labelled the rows: accuracy on rows a person labelled, agreement with an agent's or LLM's labels; go/no-go; `save` writes `models/<name>-vN` |
| `model_info(path)` | what a saved model folder holds and how to load it |

`golden.csv` has the same columns as one an LLM labelled (`id`, `text`, one per field, `split`, `labelled_by`) plus
`checked` (`agreed`, `disagreed: <fields>`, or blank when not re-checked). `labelled_by` is `agent:<name>`,
`agent:<name>+human` (your labels from the log cover some fields), `agent:<name>:synthetic` or `human`.

## Options people change

| option | default | change it when |
|---|---|---|
| `n` | 500 (Python, CLI), 200 (`golden_start`) | you want more or fewer rows labelled |
| `strategy` | `uncertain` (Python, CLI), `diverse` (`golden_start`) | `uncertain` needs the student's confidences (a log); `diverse` and `random` work on any file |
| `test` | `0.2` | you want a bigger or smaller held-out share |
| `agent` | `coding-agent` | name the labeller, e.g. `claude-code`, `codex`, so `labelled_by` says who it was |

## What can go wrong

| problem | fix |
|---|---|
| `already holds a labelling session` | finish it (`decisionsmith golden --finish ...`) or start again with `--overwrite` |
| an answer comes back in `rejected` | the error names the options; send it again, or skip the text |
| `nothing labelled yet` on finish | label at least one batch first |
| `already answered` | that id has its answer; a retry is not needed. Call `golden_batch` for what is still open |
| `... is finished` | the session already wrote `golden.csv`; edit that file, or start a new session with `overwrite=True` |
| `shares most of its words with a held-out test text` | write a different example; copies of test texts would make evaluation look better than it is |
| evaluation looks too good | against agent labels the numbers are agreement with the agent, not accuracy: check a sample by hand, and fill the rows the two answers disagreed on |

Whether an AI's outputs may be used to train a model depends on that provider's terms and your plan (API or
subscription). Check them before you train on agent- or LLM-labelled data.

**Next:** [evaluate.md](evaluate.md) for reading the report, [save-and-load.md](save-and-load.md) for using the model.
