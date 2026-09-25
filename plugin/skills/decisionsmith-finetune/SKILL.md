---
name: decisionsmith-finetune
description: "Fine-tune Laya on a harness log or labelled data with decisionsmith, locally or on Kaggle/Colab, and read the go/no-go report. Use when improving the student or when status says ready to finetune."
---
# Fine-tune Laya
1. Try `h.adapt()` first (seconds; calibration + thresholds).
2. From a harness: `h.finetune()` (needs ≥ 50 labelled decisions; switches the student only if `go`).
   From a file: `decisionsmith finetune data.csv --schema app.py:Model --out runs/v1 --json`.
3. Under 1,000 rows it trains the head only (fine on a laptop). For full training use
   `examples/finetune_kaggle_colab.ipynb` on a GPU; `h.export("train.jsonl")` gets the data there.
4. Read `report.json`: base vs fine-tuned per field, and every failed go/no-go rule in `reasons`. Never claim
   improvement without it. If `go` is false, relay the reasons (usually: more data, rare options, calibration).
5. Use the result with `student="laya:./runs/v1"`, then re-bench.

## Rules
- Decision models fail zero-shot: start new fields in `mode="shadow"` (the teacher answers) until `status()` says ready.
- Ask the user before using a paid teacher, sending data to a hosted engine (Jev, LLM APIs), or starting a long training run.
- Show `status()`, `bench` or finetune report numbers before claiming the student is good; include where it loses.
- Never change a field's mode without the user's approval, and only when `status()` advice says it is ready.
- Jev cannot be fine-tuned; to improve a Jev student use `h.adapt()`; to improve Laya use `h.adapt()` then `h.finetune()`.
- Built on Laya (Apache-2.0, Nandakishor M / Convai Innovations). Not affiliated with TypeSafe AI or Convai.
