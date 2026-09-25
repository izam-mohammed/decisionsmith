# Using decisionsmith from Claude Code, Codex and other agent CLIs

> **TL;DR** — Three ways in, pick any: (1) **MCP server** `uvx decisionsmith mcp` — works in every MCP client;
> (2) **Claude Code plugin** with skills + slash commands; (3) **plain CLI with `--json`** + `AGENTS.md`
> for agents without MCP. The agent can set up engines, bench them, run the harness, read status,
> fine-tune Laya and wire decisionsmith into your code — asking you before anything costs money or sends data out.

## 1. What an agent can do with decisionsmith

| task | how |
|---|---|
| Find LLM calls in your repo that are really classifiers | `/decisionsmith:find` (Claude Code) or the `find` prompt below |
| Turn one of them into a harness (same schema, LLM stays as teacher) | skill `decisionsmith-migrate` |
| Connect engines (LLM, Jev, Laya, systemone URL) and check they work | MCP `engines_check` / `decisionsmith doctor --json` |
| Compare engines on your labelled data | MCP `bench` / `decisionsmith bench --json` |
| Read where each field stands and what to do next | MCP `status` / `decisionsmith status --json` |
| Move a field to the next mode when advice says it's ready | skill `decisionsmith-graduate` (edits `mode=` with your approval) |
| Fine-tune Laya on logged or labelled data | MCP `finetune` / `decisionsmith finetune --json` |
| Export teacher/human answers (to train on Kaggle/Colab) | MCP `export` / `decisionsmith export --json` |
| Record human corrections | MCP `label` |

## 2. MCP server (works with every MCP client)

Run: `uvx decisionsmith mcp` (stdio). Tools (typed JSON in/out):

| tool | input | output |
|---|---|---|
| `decide` | `text`, `schema` (JSON Schema or `module:Model`), `teacher?`, `student?`, `mode?` | value per field, source, confidence, id |
| `bench` | `data` path, `schema`, `engines[]` | per-engine, per-field accuracy, ECE, latency, cost |
| `status` | `log` path | per-field agreement, sure-rate, accuracy-when-sure, advice |
| `finetune` | `data` path or `log`, `schema`, `base?`, `train?` (`auto`/`head`/`full`), `out?` | report: base vs fine-tuned metrics, `go`, checkpoint path |
| `export` | `log`, `out` | rows written, format |
| `label` | `log`, `id`, `fields` | ok |
| `engines_check` | `engines[]` | reachable?, auth ok?, latency, error + fix |
| `schema_compile` | `schema` | Jev-format questions + warnings (e.g. > 20 options) |

Client setup (verify paths against each client's current docs at release):

| client | config |
|---|---|
| **Claude Code** | `claude mcp add decisionsmith -- uvx decisionsmith mcp` |
| **Claude Desktop** | `claude_desktop_config.json` → `{"mcpServers": {"decisionsmith": {"command": "uvx", "args": ["decisionsmith", "mcp"]}}}` (or one-click `decisionsmith.mcpb`) |
| **OpenAI Codex CLI** | `~/.codex/config.toml` → `[mcp_servers.decisionsmith]` `command = "uvx"` `args = ["decisionsmith", "mcp"]` |
| **Gemini CLI** | `~/.gemini/settings.json` → `"mcpServers": {"decisionsmith": {"command": "uvx", "args": ["decisionsmith","mcp"]}}` |
| **Cursor** | `.cursor/mcp.json` → same `mcpServers` block |
| **VS Code (Copilot agent)** | `.vscode/mcp.json` → `"servers": {"decisionsmith": {"command": "uvx", "args": ["decisionsmith","mcp"]}}` |
| **Windsurf** | `~/.codeium/windsurf/mcp_config.json` → `mcpServers` block |
| **Zed** | settings → `"context_servers"` entry with the same command |
| **Cline / Continue / Goose / OpenCode** | add a stdio server with command `uvx decisionsmith mcp` |

Secrets (`TYPESAFE_API_KEY`, LLM keys) are read from the environment, never passed through tool arguments.

## 3. Claude Code plugin

Install:
```
/plugin marketplace add izam-mohammed/decisionsmith
/plugin install decisionsmith@decisionsmith
```
Contents: `.mcp.json` (the MCP server above) + skills + commands + subagents.

### Skills
| skill | loads when | teaches Claude to |
|---|---|---|
| `decisionsmith` | user needs classify / route / screen / score / yes-no in code | use `ds.harness` with a Pydantic schema; pick teacher/student; start in `shadow` or `cascade`; never trust a zero-shot student blindly |
| `decisionsmith-setup` | connecting Jev, Laya, an LLM or a systemone URL | set env vars, run `engines_check`, explain costs and privacy |
| `decisionsmith-migrate` | an LLM prompt whose output is a label/yes-no/score | keep the LLM as teacher, add the student in `shadow`, same schema, add a test |
| `decisionsmith-graduate` | "can we drop the LLM?", status questions | read `status`, move fields shadow → cascade → student only when advice says ready, with approval |
| `decisionsmith-bench` | "which is better, LLM, Jev or Laya?" | run `bench` on labelled data, report honestly incl. where each loses |
| `decisionsmith-finetune` | improving the student, "train Laya on this" | `adapt` first; then `finetune` (head-only on small data, Kaggle/Colab for full), read the report, point `student="laya:./runs/v1"`, re-bench |

Core skill (draft `skills/decisionsmith/SKILL.md`):
````markdown
---
name: decisionsmith
description: Add fast, calibrated decisions (classify, route, screen, score, yes/no) to code with the
  decisionsmith harness — an LLM or Jev as teacher, Laya as a fast student, one Pydantic schema. Use when the
  answer is one of a known set of options. Not for generating text.
---
# decisionsmith
Built on Laya (Apache-2.0, Nandakishor M / Convai Innovations). Not affiliated with TypeSafe.

## Default pattern
```python
import decisionsmith as ds
h = ds.harness(Ticket, teacher="claude-sonnet-5", student="laya", mode="shadow")
ticket = h(text)
```
## Rules
- Decision models fail zero-shot: start with `mode="shadow"` (teacher answers) until `h.status()` says ready.
- Ask before using a paid teacher or sending data to a hosted engine (Jev, LLM APIs).
- Show `status`/`bench` numbers before claiming the student is good.
- Jev cannot be fine-tuned; to improve the student, `h.adapt()` then `h.finetune()` on Laya.
- Never claim a fine-tuned model is better without its held-out report (base vs fine-tuned).
````

### Slash commands
| command | does |
|---|---|
| `/decisionsmith:setup` | connect and check engines |
| `/decisionsmith:find` | list LLM calls that are really classifiers, with a proposed schema each (read-only) |
| `/decisionsmith:migrate <file:line>` | convert one to a harness in shadow mode + a test |
| `/decisionsmith:bench <data>` | compare engines |
| `/decisionsmith:status` | where each field stands + next step |
| `/decisionsmith:graduate` | propose mode changes backed by status numbers |
| `/decisionsmith:finetune` | fine-tune Laya (local or Kaggle/Colab) and report |

### Subagents
`classifier-finder` (read-only repo scan) · `bench-analyst` (runs bench, explains results) ·
`label-reviewer` (spots teacher/human disagreements to label).

### Optional hooks (later, opt-in plugin `decisionsmith-guard`)
From the earlier plan: a `PreToolUse` guard for risky shell commands and a `PostToolUse` injection check,
powered by a warm local harness daemon, fail-open. See `reference/earlier-specs/05-agents-and-claude-kit.md` § 7.

## 4. Codex, and any agent without MCP

- `AGENTS.md` at the repo root (and a snippet users can paste into theirs) describes: when to use
  decisionsmith, the default pattern, the CLI commands with `--json`, and the approval rules.
- Every CLI command: `--json`, exit codes `0 ok · 1 error · 2 unsure/advice pending · 3 invalid input ·
  4 engine unavailable`, never prompts without a TTY, errors carry `code`, `fix`, `docs`.
- `llms.txt` + `llms-full.txt` on the docs site; docstrings with runnable examples; `py.typed`.
- Codex/Cursor/Gemini users get the same capabilities through MCP (§ 2) or the CLI.

## 5. Other surfaces

| surface | how |
|---|---|
| Claude.ai / Claude Desktop | upload the skills as a zip; `.mcpb` for one-click MCP |
| Claude API / Claude Agent SDK | use the skills; harness as a tool via `ds.as_tool(Model, format="anthropic")`; SDK hooks example |
| OpenAI Agents SDK | `@input_guardrail` backed by a harness (see integrations.md) |
| Cursor / Windsurf rules | ready-made rule files pointing at the default pattern |

## 6. Tests for the agent surface
- MCP tool JSON schemas validated; each tool tested with FakeEngine.
- CLI `--json` golden outputs and exit codes.
- Plugin manifest + SKILL.md front-matter validation.
- Skill trigger tests (prompts that should / shouldn't load each skill).
- Agent CI (weekly): Claude Code and Codex complete tasks from docs only — "migrate this LLM classifier to a
  harness in shadow mode with a test", "bench LLM vs Laya on examples/data/toy.csv and summarise".
