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

## Playbook: build a model (Codex, Gemini CLI, Cursor, any MCP client)

You label the data yourself; no LLM API key is needed. Connect the MCP server (`uvx decisionsmith mcp`, see
`docs/agents.md` for each client's config), then follow these steps and stop for the user's approval where marked.
Full page: `docs/agent-build.md`.

1. Read a sample of the texts and run `data_check(path)`. Propose labels (or a Pydantic schema) with a one-line
   description per option. **Ask the user to approve.**
2. `golden_start(path, labels=[...], n=200, strategy="diverse", agent="<your name, e.g. codex>")` writes
   `golden.session.json`. Without MCP: `decisionsmith golden texts.csv --labels a,b,c --teacher agent:codex
   --strategy diverse --json`.
3. Loop `golden_batch(session)` then `golden_submit(session, [{"id": ..., "answers": {field: option}}], agent=...)`
   until `items` is empty. Skip a text you are unsure of with `{"id": ..., "skip": "why"}`. When `pass` is 2 (the
   blind re-check), label again without reading your earlier answers (a fresh session or subagent if you can).
4. `golden_status(session)`: show the user the disagreements and skipped rows. If an option has fewer than 10 rows,
   ask, then `golden_add(session, [{"text": ..., "answers": {...}}])` (marked synthetic, re-checked blind).
5. `golden_finish(session)` (or `decisionsmith golden --finish golden.session.json`), then
   `finetune("golden.csv", session, out="runs/v1")` (ask before long runs) and
   `evaluate("runs/v1", "golden.csv", session, save="models/<name>")`. Show the report, the accuracy by who labelled
   and every go/no-go reason. **Ask the user to approve.**
6. Write the production code (`ds.load("models/<name>-v1")` and `ds.harness(model, teacher=..., mode="shadow",
   collect=0.05)`). **Ask the user to approve the diff.**

Honesty: quote only measured numbers; accuracy against your own labels is agreement with you, not truth; flag
synthetic rows; tell the user that using AI outputs to train a model depends on the provider's terms and their plan.

Headless (scripts, CI): `claude -p "/decisionsmith:build tickets.csv --labels billing,technical,sales"` runs the same
flow in Claude Code with the plugin installed. Nobody is there to approve, so the labels you pass count as step 1's
approval; add "stop after the report" to the prompt to skip step 6, and review the report and any diff it leaves.
