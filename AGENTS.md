# AGENTS.md: using decisionsmith

decisionsmith adds fast, calibrated decisions (classify, route, screen, score, yes/no) with an LLM or Jev as the
teacher and Laya as a fast student, behind one Pydantic schema. Use it when the answer is one of a known set of
options; not for generating text.

```python
import decisionsmith as ds

h = ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya", mode="shadow")
h(text)  # -> Ticket
h.status()  # per field: agreement, sure rate, accuracy when sure, advice
h.adapt()  # calibration + thresholds
h.finetune()  # train Laya on the log; switches only if better
```

CLI (every command takes `--json`; exit codes 0 ok · 1 error · 2 done but not ready · 3 invalid input · 4 engine
unavailable; never prompts):

| command | does |
|---|---|
| `decisionsmith doctor [--engines a,b]` | installs, device, keys, engine checks |
| `decisionsmith bench data.csv --schema app.py:M --engines a,b` | compare engines |
| `decisionsmith status --schema app.py:M --log decisions.db` | where each field stands |
| `decisionsmith export --schema app.py:M --log decisions.db --out train.jsonl` | training data from the log |
| `decisionsmith finetune data.csv --schema app.py:M --out runs/v1` | fine-tune Laya; report + go/no-go |
| `decisionsmith mcp` | MCP server with the same tools |

Rules for agents:
- Start new fields in `shadow` mode; change modes only with the user's approval and only when `status()` says ready.
- Ask before paid engines, hosted engines (data leaves the machine) and long training runs.
- Quote `status` / `bench` / report numbers, including losses; never claim accuracy without them.
- Secrets come from the environment (`TYPESAFE_API_KEY`, provider keys); never put them in code or arguments.
