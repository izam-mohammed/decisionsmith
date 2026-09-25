# Changelog

## Unreleased

- **LLM teachers need no LLM library.** A built-in OpenAI-compatible client on httpx (OpenAI, Gemini, Groq,
  OpenRouter, Together, Fireworks, DeepSeek, xAI, Mistral, Ollama, any `ds.LLM(model, url=...)`), and Claude through
  the official `anthropic` SDK (`decisionsmith[anthropic]`). Core dependencies stay `pydantic` + `httpx`.
- **Breaking: the `[llm]` extra is gone.** LiteLLM and Instructor are no longer used by the core. Model strings
  such as `"gpt-5"`, `"claude-sonnet-5"` and `"ollama/qwen3"` keep working without them; for anything else LiteLLM
  supports, install `decisionsmith[litellm]` and use `"litellm/<id>"`. An unknown model string without a provider
  prefix now raises a `ValueError` naming the fix. `bench` reports cost only for LiteLLM teachers.
- Framework LLM objects as teachers: LangChain, LlamaIndex, DSPy, CrewAI, Pydantic AI, Haystack, smolagents,
  AutoGen and Semantic Kernel objects are detected and wrapped (`decisionsmith.integrations.<name>`, one extra each).
- `decisionsmith.integrations.litellm`: LiteLLM teacher, Proxy guardrail, Router tier picker.
- Async: `h.adecide()`, `h.acall()`, `model.apredict()` (native async for HTTP engines).
- `DS_OFFLINE=1` / `DS_LAYA=<dir>`: run anything without keys or network, for examples, notebooks and demos.
- Examples move to `examples/<group>/<name>/` with a generated README each and a generated gallery; CI runs every
  example offline.

## 0.1.0 (unreleased)

First release: the harness (teacher + student, four modes per field, SQLite log, `status`, `label`, `adapt`,
`export`), Laya fine-tuning (`ds.finetune`, `h.finetune()`, `decisionsmith finetune`: head-only or full, held-out
calibration, Laya-compatible checkpoints, report with go/no-go, DDP), `bench`, the CLI, the MCP server, the Claude
Code plugin and AGENTS.md. Decisions: DECISIONS.md H1 to H31.
