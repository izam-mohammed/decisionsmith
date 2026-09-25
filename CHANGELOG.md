# Changelog

## 0.1.0 (25/09/2026)

- **Notebooks** in `notebooks/`: quickstart, golden dataset, harness shadow to cascade, full GPU fine-tune,
  multilingual heads, typed-decisions reproduction and bench. The laptop ones keep outputs from real runs; CI runs
  every notebook offline. The Kaggle/Colab notebook moved from `examples/finetune_kaggle_colab.ipynb` to
  `notebooks/04_finetune_full_gpu.ipynb`.
- **Saved models, review fixes.** `model.train()` after `evaluate()` forgets the old report and thresholds; `ds.load`
  with a class compares every question (descriptions and docstring too) and lists the differences; versions are
  numbered one above the highest; a loaded model saves as the next version of its own name; a labels model is named
  after its labels; `save` writes atomically, retries the next number on a clash, expands `~` and prints
  `saved to ...`; saved reports hold no texts, row ids or local paths; `Literal` int values round-trip; the model
  card no longer claims a licence for your model; `ds.model(labels, saved_folder)` loads it like `ds.load`;
  `Model.info` is now `Model.meta`.
- **`evaluate` picks thresholds on one half of the rows and reports coverage on the other**, and adds
  `accuracy_when_sure`.
- **An explicit `threshold=` on `ds.harness` wins over a loaded model's saved thresholds** (`h.adapt()` results on
  the log still come first).
- **Docs are tested:** every Python block in README.md and docs/ runs offline in the test suite.
- **`ds.golden(source, teacher, n=500, strategy="uncertain")`** picks texts from a harness log, a file or a list
  (`uncertain`, `disagree`, `diverse`, `random`), labels them with the main LLM and writes `golden.csv` with a
  `split` column (`train`, `calib`, `test`; `dev`/`val` mean `calib`) and `labelled_by` (`human`, `llm:<model>`).
  Human labels in the log are used as they are; an existing file is never overwritten without `overwrite=True`. Training never sees `split=test` rows and fits
  calibration on `split=calib` rows; `evaluate` uses only `split=test` rows. Trained models record one-way
  fingerprints of their training texts, and `evaluate` flags (no-go) any test text the model trained on.
- **`collect=` on the harness** keeps the text of a share of decisions plus every unsure or disputed one
  (`collect=0` stores no text); **`h.forget(id)`** and **`h.forget(older_than_days=30)`** delete decisions and
  overwrite their bytes in the log.
- **`ds.bench` on a file with a `split` column uses only the `split=test` rows.** A blank split counts as `train`
  whenever any row has a split.
- **CLI `decisionsmith golden`** (from `--log` or a texts file) and **`decisionsmith eval`** (a saved model on labelled
  data; exit code 2 on no-go).
- **Security: resume state is no longer pickled.** `checkpoint_latest/` now holds `state.safetensors` + `state.json`
  (weights, optimizer, schedule, RNG); the old `state.pt` (`torch.load(weights_only=False)`) is refused, never
  loaded. Training reaches laya's question normalisation through one checked shim with a clear error if laya changes it.
- **Saved models are versioned folders.** `model.save()` writes `models/<name>-vN` (never overwrites a version)
  with the Laya checkpoint, `decisionsmith.json` (labels or schema, calibration, thresholds, provenance),
  `report.json` and `MODEL_CARD.md`. **Breaking:** `model.save("x")` now writes `x-v1`, then `x-v2`.
- **`ds.load(path)`** returns a ready model with its labels or schema, calibration and thresholds; pass your own
  class to get it back (it must match). A harness built from a loaded model uses its thresholds.
- **`model.evaluate(data)`**: per field accuracy, macro-F1, ECE, threshold, coverage, confusions, worst cases and
  latency, with go/no-go reasons.
- **LLM teachers need no LLM library.** A built-in OpenAI-compatible client on httpx (OpenAI, Gemini, Groq,
  OpenRouter, Together, Fireworks, DeepSeek, xAI, Mistral, Ollama, any `ds.LLM(model, url=...)`), and Claude through
  the official `anthropic` SDK (`decisionsmith[anthropic]`). Core dependencies stay `pydantic` + `httpx`.
- **Breaking: the `[llm]` extra (LiteLLM + Instructor) is gone, and LLM teachers need no extra now.** LiteLLM and
  Instructor are no longer used by the core. The simonw/llm plugin is `[llm-plugin]`, not `[llm]`. Model strings
  such as `"gpt-5"`, `"claude-sonnet-5"` and `"ollama/qwen3"` keep working without them; for anything else LiteLLM
  supports, install `decisionsmith[litellm]` and use `"litellm/<id>"`. An unknown model string without a provider
  prefix now raises a `ValueError` naming the fix. `bench` reports cost only for LiteLLM teachers.
- Framework LLM objects as teachers: LangChain, LlamaIndex, DSPy, CrewAI, Pydantic AI, Haystack, smolagents,
  AutoGen and Semantic Kernel objects are detected and wrapped (`decisionsmith.integrations.<name>`, one extra each).
- `decisionsmith.integrations.litellm`: LiteLLM teacher, Proxy guardrail, Router tier picker.
- Gateways and structured output: `integrations.openai` (SDK client teacher, `wrap(client, x)`),
  `integrations.instructor` (`wrap`), `integrations.portkey` (webhook guardrail), `integrations.llm_plugin`
  (`llm decide` command, `llm` models as teachers), `integrations.outlines` (Outlines models as teachers) and
  `integrations.marvin` (`classify` drop-in), with examples for every built-in and LiteLLM provider.
- Agent frameworks: `integrations.langchain` (`DecisionRunnable`, `as_tool`, `output_parser`, `compressor`),
  `integrations.langgraph` (`route_on`, `guard_node`), `integrations.pydantic_ai` (`tool`, `output_validator`,
  `router`), `integrations.openai_agents` (input / output guardrails, `function_tool`, `router`, model teacher),
  `integrations.claude_agent_sdk` (`PreToolUse` / `PostToolUse` hooks), `integrations.llamaindex`
  (`relevance_filter`, `selector`, `as_tool`) and `integrations.google_adk` (`tool`, `before_model_callback` guard),
  with teacher examples for every provider each framework supports.
- More agent frameworks: `integrations.crewai` (`tool`, task `guardrail`), `integrations.dspy` (`DecisionModule`,
  `metric`), `integrations.haystack` (`router`, `document_filter` components), `integrations.autogen` (`tool`,
  `stop_on` termination), `integrations.semantic_kernel` (`plugin`, `invocation_filter`), `integrations.smolagents`
  (`tool`) and the new `integrations.agno` (`tool`, Agno models as teachers), with teacher examples per provider.
- Async: `h.adecide()`, `h.acall()`, `model.apredict()` (native async for HTTP engines).
- `DS_OFFLINE=1` / `DS_LAYA=<dir>`: run anything without keys or network, for examples, notebooks and demos.
- Examples move to `examples/<group>/<name>/` with a generated README each and a generated gallery; CI runs every
  example offline.

## 0.1.0 (unreleased)

First release: the harness (teacher + student, four modes per field, SQLite log, `status`, `label`, `adapt`,
`export`), Laya fine-tuning (`ds.finetune`, `h.finetune()`, `decisionsmith finetune`: head-only or full, held-out
calibration, Laya-compatible checkpoints, report with go/no-go, DDP), `bench`, the CLI, the MCP server, the Claude
Code plugin and AGENTS.md.
