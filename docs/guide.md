# Guide: engines, modes, status, adapt, bench

## Engines

| string | engine | notes |
|---|---|---|
| `"claude-sonnet-5"`, `"gpt-5"`, `"gemini-2.5-flash"`, `"groq/<m>"`, `"ollama/qwen3"`, `ds.LLM(m, url=...)`, `"litellm/<id>"`, a LangChain / LlamaIndex / DSPy / ... LLM object | LLM | built-in client, no LLM library; one call answers every field; text is fenced as data; answers are one-hot ([teachers.md](teachers.md)) |
| `"jev"`, `"jev:<model>"` | TypeSafe Jev | `TYPESAFE_API_KEY`; default model `jev-1.13.0` |
| `"laya"`, `"laya:multilingual"`, `"laya:typed-decisions"`, `"laya:./runs/v1"`, `"laya:org/repo"` | Laya in-process | loaded once per process, batched in `h.many()` |
| `"systemone:http://host:8000"` | any Jev-compatible server | `SYSTEMONE_API_KEY` as bearer token; retries 408/429/5xx |
| `"fake"` | chance baseline | tests and demos |

Custom engine: any object with `name: str` and `ask(text, questions) -> {"answers": {...}}` in the Jev format
(add `async aask(...)` to make `h.adecide()` native async; without it, it runs in a thread).
`decisionsmith doctor --engines laya,claude-haiku-4-5` checks installs, keys and engines.

## Modes

| mode | answers | also runs |
|---|---|---|
| `teacher` | teacher | nothing |
| `shadow` | teacher | student, in parallel, logged |
| `cascade` | student if confidence ≥ threshold, else teacher | teacher on 5% of sure answers (`audit=`), logged |
| `student` | student | teacher only if the student fails |

`mode="shadow"` for all fields or `mode={"team": "cascade"}` per field. Fields you don't name get the default
(`cascade` with two engines). `threshold=0.8` is the default; `adapt()` replaces it per field.
`h.decide(text)` returns a `ds.Result`: `.value`, `.source`, `.confidence`, `.probabilities`, `.sure`, `.id`.

## Status

Per field, for the current student: `agreement` with the teacher, `sure_rate`, `accuracy_when_sure` (against
human labels once there are 30 of them, else against the teacher), labelled rows, and one line of advice:

| advice | rule |
|---|---|
| ready for student | ≥ 500 rows where both ran, agreement ≥ 97%, accuracy when sure ≥ 95% |
| ready for cascade | ≥ 200 rows where both ran, agreement ≥ 90%, accuracy when sure ≥ 95% |
| ready to finetune | ≥ 300 labelled rows, ≥ 20 per option |
| student disagrees often | ≥ 200 rows where both ran, below the cascade rule |
| needs more data | anything else |

CLI: `decisionsmith status --schema app.py:Ticket --log decisions.db`.

## Adapt

`h.adapt(target=0.97)` needs 100 labelled rows per field for the current student. It fits a temperature on half
of them and picks, on the other half, the smallest threshold whose sure answers reach the target accuracy. It
never changes an answer; it changes confidence and routing. A field with no threshold that reaches the target
always goes to the teacher. The report shows ECE before and after, coverage, and the most confused option pair
(rewrite those `ds.Options` descriptions first). Works the same for Jev and Laya. Rows used for fine-tuning are
skipped.

## Bench

`ds.bench(Ticket, "labelled.csv", ["claude-haiku-4-5", "jev", "laya", "laya:./runs/v1"])` runs each engine
zero-shot on the same rows: accuracy, macro-F1, ECE, coverage and accuracy at the threshold, latency p50/p95, and
cost per 1k decisions (from reported usage; Jev from its list price; Laya counts as 0, hardware not included).

## The log

`decisions.db` (SQLite) holds every input text and both engines' probabilities, because fine-tuning needs them.
Keep it where you keep other customer data. `log=None` turns logging off (and with it status, adapt and
finetune).
