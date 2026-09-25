# Fine-tuning Laya

> **TL;DR**: `decisionsmith finetune data.csv --schema app.py:Ticket` (or `h.finetune()` from a harness log)
> writes a plain Laya checkpoint, calibrated on held-out data, with a report and a go/no-go verdict.
> Head-only training below 1,000 rows runs on a laptop; full fine-tuning wants a GPU (Kaggle/Colab notebook).

```bash
decisionsmith finetune tickets.csv --schema app.py:Ticket --out runs/v1
```
```python
report = ds.finetune("tickets.csv", Ticket, out="runs/v1")
print(report)  # base vs fine-tuned on the test split, go/no-go, reasons
h = ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya:./runs/v1")
```

## Data

| format | shape |
|---|---|
| CSV | `text` + one column per field (`billing`, `true`/`false`/`yes`/`no`, a level); blank = unlabelled |
| JSONL, answers | `{"text": ..., "answers": {"team": "sales", "wants_refund": {"true": 0.8, "false": 0.2}}}` |
| JSONL, typed-decisions | Laya's notebook format `{state, questions, gold}`; no schema needed |
| harness log | `h.finetune()` reads it directly; `h.export("train.jsonl")` writes it for a remote GPU |

Errors name the file line and field. Rows are split train / calib (min(400, 10%)) / test (15%) with a fixed seed;
`--group-by customer_id` keeps related rows on one side.

## What trains

| `--train` | what | where | default when |
|---|---|---|---|
| `head` | the decision head only; the encoder is frozen | CPU, Apple Silicon, any GPU | fewer than 1,000 training rows |
| `full` | the whole model | a GPU | 1,000 rows or more |

Measured on this repo's 300-row toy set with the real Laya English checkpoint, head-only, on an Apple Silicon Mac:
about 1 minute end to end, 23 ms per decision. That's one small, easy dataset; your numbers will differ.

Options: `--epochs 4 --batch 8 --accum --lr --loss ce|proper|rlcd --seed 0 --device --resume --max-steps`.
Training stops early when the calibration loss stops improving and keeps the best epoch. `--resume` picks up
after the last finished epoch (same `--out`, same settings).

## Output (`runs/v1/`)

| file | |
|---|---|
| `rl_agent_config.json` `model.safetensors` `tokenizer/` `encoder/` | the Laya checkpoint; `laya.load("runs/v1")` works |
| `report.json` `report.html` | base vs fine-tuned per field: accuracy, macro-F1, ECE, coverage curve, worst examples |
| `MODEL_CARD.md` | base model, licence (Apache-2.0), credits |
| `train_log.jsonl` | loss per epoch |
| `checkpoint_latest/` | resume state (delete it before sharing) |

Temperatures are fitted on the calibration split only, per question type (and per option-count bucket with
2,000+ samples), clamped to Laya's 0.5 to 5.0.

## Go / no-go

`go` is true only when all of these hold; every failed one is listed in `report.reasons`:
- the test split has at least 100 decisions, and every option has at least 10 test rows
- calibration error (ECE) is at most 0.10
- fine-tuned accuracy beats the base, and no field drops more than 2 points
- the saved checkpoint loads in `laya.load`

`h.finetune()` switches the harness to the new checkpoint only when `go` is true. Modes never change by themselves.

## On a GPU (Kaggle or Colab)

1. `h.export("train.jsonl")` (or use your CSV).
2. Open `examples/finetune_kaggle_colab.ipynb` on Kaggle (GPU T4 x2) or Colab (T4), add the file, run all.
   With two GPUs it trains data-parallel (`torchrun -m decisionsmith.cli finetune ...`).
3. Download `runs/v1.zip`, unzip, `student="laya:./runs/v1"`.

## Multilingual

`--base laya:multilingual` fine-tunes the multilingual checkpoint; `--train head` trains new decision heads on the
frozen multilingual encoder (the laya#320 ask).
