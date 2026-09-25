---
name: decisionsmith-bench
description: "Compare LLMs, Jev and Laya (base or fine-tuned) on the user's labelled data with decisionsmith bench. Use when asked which engine is better, faster or cheaper."
---
# Bench engines
1. Needs labelled data: CSV with `text` + one column per field, or answers JSONL.
2. `decisionsmith bench data.csv --schema app.py:Model --engines claude-haiku-4-5,jev,laya --json`
   (ask before paid engines; `--limit 100` for a cheap first pass).
3. Report per field: accuracy, macro-F1, ECE, coverage at threshold, p50 latency, cost per 1k. Say where each engine
   loses, and that this is zero-shot unless a `laya:./runs/...` checkpoint is included.

## Rules
- Decision models fail zero-shot: start new fields in `mode="shadow"` (the teacher answers) until `status()` says ready.
- Ask the user before using a paid teacher, sending data to a hosted engine (Jev, LLM APIs), or starting a long training run.
- Show `status()`, `bench` or finetune report numbers before claiming the student is good; include where it loses.
- Never change a field's mode without the user's approval, and only when `status()` advice says it is ready.
- Jev cannot be fine-tuned; to improve a Jev student use `h.adapt()`; to improve Laya use `h.adapt()` then `h.finetune()`.
- Built on Laya (Apache-2.0, Nandakishor M / Convai Innovations). Not affiliated with TypeSafe AI or Convai.
