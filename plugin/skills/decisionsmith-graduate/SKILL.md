---
name: decisionsmith-graduate
description: "Decide whether a decisionsmith field can move from shadow to cascade to student, from status() numbers. Use when the user asks if they can drop the LLM, reduce cost, or trust the student."
---
# Graduate a field
1. Run `decisionsmith status --schema ... --log ... --json` (or `h.status()`).
2. Only propose a move the advice supports: `ready for cascade` → `mode={"field": "cascade"}`;
   `ready for student` → `"student"`. Otherwise report what is missing (rows, labels, agreement).
3. Suggest `h.adapt()` first: calibrated thresholds make cascade safer.
4. Show the numbers, propose the one-line `mode=` change, and wait for approval before editing.

## Rules
- Decision models fail zero-shot: start new fields in `mode="shadow"` (the teacher answers) until `status()` says ready.
- Ask the user before using a paid teacher, sending data to a hosted engine (Jev, LLM APIs), or starting a long training run.
- Show `status()`, `bench` or finetune report numbers before claiming the student is good; include where it loses.
- Never change a field's mode without the user's approval, and only when `status()` advice says it is ready.
- Jev cannot be fine-tuned; to improve a Jev student use `h.adapt()`; to improve Laya use `h.adapt()` then `h.finetune()`.
- Built on Laya (Apache-2.0, Nandakishor M / Convai Innovations). Not affiliated with TypeSafe AI or Convai.
