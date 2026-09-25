---
name: decisionsmith-migrate
description: "Turn an existing LLM call whose output is a label, yes/no or score into a decisionsmith harness in shadow mode, keeping the LLM as teacher. Use when code prompts an LLM to classify, route, moderate or grade."
---
# Migrate an LLM classifier
1. Find the call and its allowed outputs; write a Pydantic model with one field per decision.
2. Replace the call with `h = ds.harness(Model, teacher="<the same model id>", student="laya", mode="shadow")` and
   `h(text)`; the LLM keeps answering, so behaviour is unchanged.
3. Keep the log path somewhere persistent (`log="data/decisions.db"`).
4. Add a test with `ds.testing.FakeEngine(truth)` as the teacher so CI needs no keys.
5. Tell the user how to check progress: `decisionsmith status --schema app.py:Model --log data/decisions.db`.

## Rules
- Decision models fail zero-shot: start new fields in `mode="shadow"` (the teacher answers) until `status()` says ready.
- Ask the user before using a paid teacher, sending data to a hosted engine (Jev, LLM APIs), or starting a long training run.
- Show `status()`, `bench` or finetune report numbers before claiming the student is good; include where it loses.
- Never change a field's mode without the user's approval, and only when `status()` advice says it is ready.
- Jev cannot be fine-tuned; to improve a Jev student use `h.adapt()`; to improve Laya use `h.adapt()` then `h.finetune()`.
- Built on Laya (Apache-2.0, Nandakishor M / Convai Innovations). Not affiliated with TypeSafe AI or Convai.
