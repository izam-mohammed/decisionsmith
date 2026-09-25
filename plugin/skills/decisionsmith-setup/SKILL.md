---
name: decisionsmith-setup
description: "Connect and check decisionsmith engines: Jev (TYPESAFE_API_KEY), Laya (local), any LLM via LiteLLM, or a systemone URL. Use when setting up keys, installing extras or when an engine fails."
---
# Set up engines
1. Install: `uv add "decisionsmith[all]"` (or `[laya]`, `[llm]`, `[mcp]`).
2. Keys come from the environment only: `TYPESAFE_API_KEY` (Jev), provider keys such as `ANTHROPIC_API_KEY` (LLMs),
   `SYSTEMONE_API_KEY` (bearer token for a systemone server). Never write keys into code or tool arguments.
3. Check: `decisionsmith doctor --engines laya,claude-haiku-4-5 --json` (one tiny request per engine; say it may cost
   a fraction of a cent before running it with paid engines).
4. Explain what leaves the machine: LLM and Jev engines send the text to that provider; Laya and Ollama stay local.

## Rules
- Decision models fail zero-shot: start new fields in `mode="shadow"` (the teacher answers) until `status()` says ready.
- Ask the user before using a paid teacher, sending data to a hosted engine (Jev, LLM APIs), or starting a long training run.
- Show `status()`, `bench` or finetune report numbers before claiming the student is good; include where it loses.
- Never change a field's mode without the user's approval, and only when `status()` advice says it is ready.
- Jev cannot be fine-tuned; to improve a Jev student use `h.adapt()`; to improve Laya use `h.adapt()` then `h.finetune()`.
- Built on Laya (Apache-2.0, Nandakishor M / Convai Innovations). Not affiliated with TypeSafe AI or Convai.
