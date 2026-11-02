# Integrations

> **TL;DR**: decisionsmith plugs in two ways. **Engines** answer (any LLM, Jev, Laya, any systemone URL). And
> **integrations** put a model or harness inside your framework, or use your framework's LLM as the teacher. Every
> integration is one module, `decisionsmith.integrations.<name>` (about 100 lines or less), behind its own extra,
> tested offline against the pinned framework version in its own CI job. `import decisionsmith` imports none of them.

## Shipped

| extra | module | what you get | tested against |
|---|---|---|---|
| none | core | built-in LLM client: OpenAI, Gemini, Groq, OpenRouter, Together, Fireworks, DeepSeek, xAI, Mistral, Ollama, any OpenAI-compatible `url` ([teachers.md](teachers.md)) | respx-mocked HTTP |
| `anthropic` | core | Claude teachers through the official SDK (`messages.parse`) | anthropic 1.8.0, mocked transport |
| `litellm` | `integrations.litellm` | `teacher="litellm/<id>"` (any LiteLLM model, with cost), `guardrail(x, field, block)` Proxy `CustomGuardrail` (pre and post call), `tier(x, field, models)` model picker for a `Router` | litellm 1.102.1 |
| `openai` | `integrations.openai` | `teacher(client, model)` for any `openai` client (any `base_url`: OpenAI, Claude's compatibility endpoint, Gemini, Groq, Ollama, ...); `wrap(client, x)`: `chat.completions.create` / `.parse` with a `response_format` that only asks for the model's fields is answered by the model when it is sure, otherwise the real call is made | openai 3.19.2 |
| `instructor` | `integrations.instructor` | `wrap(client, x)`: an Instructor client whose `create(response_model=...)` is answered by the model when it is sure, else by Instructor's LLM call | instructor 1.17.0, openai 2.54.0 |
| none | `integrations.portkey` | `webhook(x, field, block)`: takes Portkey's webhook guardrail request JSON (`beforeRequestHook` / `afterRequestHook`), returns `{"verdict": ...}`; sync or `await check.acall(body)` | Portkey's documented webhook format |
| `llm-plugin` | `integrations.llm_plugin` | an `llm decide` command for simonw's `llm` CLI (registered through the `llm` entry point) and any `llm` model as the teacher (detected automatically) | llm 0.36 |
| `outlines` | `integrations.outlines` | any Outlines model (transformers, llama.cpp, MLX, vLLM, Ollama, OpenAI, ...) as the teacher, with the answer schema enforced by structured generation (detected automatically) | outlines 1.3.3 (Python < 3.14), OpenAI-backed model on a mocked transport |
| none | `integrations.marvin` | `classify(data, labels, ...)` and `classify_async` with Marvin's signature, answered by a `ds.model` (base Laya by default, or `model=`); `multi_label`, `agent`, `thread`, `context`, `handlers` and `prompt` raise | signature checked against marvin 3.2.7 |
| `langchain` | `integrations.langchain` | any LangChain chat model as the teacher; `DecisionRunnable(x, field)` (a `Runnable`: `invoke` / `ainvoke` / `batch`, composes with `\|`, str / message / dict input); `as_tool(x, name, description)` (`StructuredTool`); `output_parser(x, field)` (LLM text to decision); `compressor(x, field, keep)` (`BaseDocumentCompressor` for RAG relevance filtering) | langchain-core 1.6.5; providers: langchain-openai 1.6.6, -anthropic 1.7.4, -google-genai 4.4.0, -aws 1.7.9, -mistralai 1.1.6, -groq 1.1.3, -ollama 1.1.0, -huggingface 1.2.2, -fireworks 1.6.2, -together 0.4.0 |
| `langgraph` | `integrations.langgraph` | `route_on(x, field, key="messages")`: a path for `add_conditional_edges` (sync and async); `guard_node(x, field, block, key, flag="blocked")`: a node that sets `state["blocked"]` when the decision is one of `block` | langgraph 1.2.12, langchain-core 1.6.5 |
| `llamaindex` | `integrations.llamaindex` | any LlamaIndex LLM as the teacher; `relevance_filter(x, field, keep)` node postprocessor for RAG (keeps nodes whose decision is in `keep`); `selector(x, field)` for `RouterQueryEngine` (picks the tool whose name is the decision); `as_tool(x, name, description)` `FunctionTool` | llama-index-core 0.14.25, llama-index-llms-openai 0.7.9 (+ anthropic, google-genai, bedrock-converse, azure-openai, mistralai, groq, ollama, huggingface-api) |
| `dspy` | `integrations.dspy` | a `dspy.LM` as the teacher (any provider DSPy reaches through LiteLLM); `DecisionModule(x, input="text")`: a `dspy.Module` whose `forward` / `aforward` return a `dspy.Prediction` with one output per field (`label` for `ds.model(labels)`); `metric(*fields)` for `dspy.Evaluate` and optimisers (the share of fields that match the example, a bool when `trace` is set) | dspy 3.4.0 (real `dspy.Evaluate`) |
| `crewai` | `integrations.crewai` | a CrewAI `LLM` as the teacher (OpenAI, Anthropic, Gemini, Ollama, Azure); `tool(x, name, description, field)`: a `BaseTool` taking `text` and returning the decision (sync and async); `guardrail(x, field, block, message)` for `Task(guardrail=...)` / `Agent(guardrail=...)`: returns `(False, feedback)` so the agent retries when the decision about the output is one of `block` | crewai 1.15.22 (Python < 3.14) with anthropic 0.73.0, google-genai 1.65.0, azure-ai-inference 1.0.0b9; real `Crew.kickoff` on a mocked OpenAI transport |
| `pydantic-ai` | `integrations.pydantic_ai` | any Pydantic AI model as the teacher; `tool(x)` (a `Tool` the agent calls with a text), `output_validator(x, field, block)` for `agent.output_validator` (retries the reply when it is blocked), `router(x, field, agents)` (the decision picks the `Agent`: `pick`, `run`, `run_sync`) | pydantic-ai-slim 2.50.0 |
| `openai-agents` | `integrations.openai_agents` | `input_guardrail(x, field, block)` / `output_guardrail(...)` (SDK guardrail objects, tripwire when blocked, the decision as `output_info`), `function_tool(x)`, `router(x, field, agents)` (hands the input to the picked `Agent` with no triage LLM call), `teacher(model)` for any Agents SDK model (OpenAI, `LitellmModel`) | openai-agents 0.22.3 (openai 3.19.2), the SDK's `ScriptedModel` |
| `claude-agent-sdk` | `integrations.claude_agent_sdk` | `hook(x, field, block, event="PreToolUse" or "PostToolUse", matcher, text=None)`: a `HookMatcher` for `ClaudeAgentOptions(hooks=...)` that denies a tool call (`permissionDecision: "deny"`) or blocks its output (`decision: "block"`) when the decision is one of `block`; otherwise the normal permission flow applies | claude-agent-sdk 0.2.159 (hook callbacks tested directly; the CLI is never started) |
| `google-adk` | `integrations.google_adk` | `tool(x, name, description)`: an ADK `FunctionTool` returning the decision as a dict; `guard(x, field, block, message)`: an async `before_model_callback` that replies with `message` instead of calling the model when the decision is one of `block` | google-adk 2.9.2 (real `InMemoryRunner` with an offline `BaseLlm`) |
| `haystack` | `integrations.haystack` | any Haystack chat generator as the teacher (OpenAI, Anthropic, Google GenAI, Ollama, Hugging Face API); `router(x, field, documents=False)`: a `@component` with one output per value of the decision (`text` in, or `documents` grouped per value); `document_filter(x, field, keep, with_query=True)`: a `@component` keeping the documents whose decision is in `keep` (RAG, after a retriever); sync and `run_async` | haystack-ai 3.2.0; anthropic-haystack 6.2.0, google-genai-haystack 6.0.0, ollama-haystack 7.0.1, huggingface-api-haystack 1.0.0 (huggingface-hub 1.33.0); real `Pipeline` |
| `smolagents` | `integrations.smolagents` | any smolagents model as the teacher (`OpenAIServerModel`, `LiteLLMModel`, `InferenceClientModel`, `TransformersModel`); `tool(x, name, description)` / `DecisionTool`: a `Tool` for `CodeAgent` / `ToolCallingAgent` taking `text` and returning the decision (a dict of fields, or the label) | smolagents 1.26.0 (real `CodeAgent` and `ToolCallingAgent` with a scripted `Model`) |
| `autogen` | `integrations.autogen` | any AutoGen 0.4+ model client as the teacher (OpenAI, Azure OpenAI, Anthropic, Ollama; AG2's `autogen` package is not supported); `tool(x, name, description)`: an `autogen_core` `FunctionTool` returning the decision; `stop_on(x, field, block, sources)`: a team `TerminationCondition` that stops the run when a chat message's decision is one of `block` (combines with `\|` / `&`) | autogen-agentchat / autogen-ext 0.7.5 (anthropic 1.8.0, ollama 0.6.2, openai 3.19.2), `ReplayChatCompletionClient` |
| `semantic-kernel` | `integrations.semantic_kernel` | any Semantic Kernel chat completion service as the teacher; `plugin(x, name, description)`: a plugin object with one `@kernel_function` (text in, the decision out; works with auto function calling); `invocation_filter(x, field, block, functions, message)` for `kernel.add_filter(FilterTypes.FUNCTION_INVOCATION, ...)`: the function is not run and `message` is its result when the decision about its text arguments is one of `block` | semantic-kernel 1.44.1 (anthropic 0.125.0, ollama 0.6.2; auto function calling on a mocked OpenAI transport) |
| `agno` | `integrations.agno` | any Agno model as the teacher (`OpenAIChat`, `Claude`, `Gemini`, `Ollama`, ...); `tool(x, name, description)`: an Agno tool (`Function`) taking `text` and returning the decision as JSON, for `agent.run` and `agent.arun` | agno 3.0.11 (openai 3.19.2, anthropic 1.8.0, google-genai 2.25.0, ollama 0.6.2); real `Agent` on a mocked OpenAI transport |
| `pandas` | `integrations.pandas` | `df.ds.decide("text", x, fields=None, prefix="")` (the accessor is added when you import the module) and `decide(df, "text", x)`: a column per field plus `<field>_confidence` and `<field>_source` (`student` / `teacher`), batched | pandas 3.0.6 |
| `polars` | `integrations.polars` | `decide_expr(x, "text", field=None, prefix="")`: one field as a column, or a `decision` struct of every field with confidence and source (`.unnest("decision")`), through `map_batches`; eager and lazy frames | polars 1.44.2 |
| `datasets` | `integrations.datasets` | `dataset.map(ds_map(x), batched=True)` adds the decision columns; `"hf:<dataset>"` / `"hf:<dataset>:<split>"` as training and evaluation data for `model.train`, `model.evaluate` and `ds.finetune` (a hub id or a local folder of CSV / JSON / Parquet; `ClassLabel` ids become names) | datasets 5.0.1 (local Parquet folders; hub downloads are the datasets library's own code and are not run offline) |
| `duckdb` | `integrations.duckdb` | `register(con, x, field=None, prefix="ds")`: SQL functions `ds_decide(text, field)`, `ds_confidence(text, field)` and `ds_is_true(text)` (the yes/no field), vectorised Arrow UDFs; each text is decided once per session (last 4096 texts) | duckdb 1.5.5, pyarrow 25.0.1 |
| `spark` | `integrations.spark` | `udf(build, field=None, prefix="")`: a `pandas_udf` (one field, or a struct of every field with confidence and source); `build` is a function returning the model or harness, run once on the driver and once per Python worker | pyspark 4.2.0 (pandas 2.3.3, pyarrow 25.0.1, Java 17), real `local[2]` session |
| `dask` | `integrations.dask` | `decide(ddf, "text", x, fields=None, prefix="")`: lazy decision columns via `map_partitions`; `x` can be a builder function for process-based schedulers | dask[dataframe] 2026.8.0 (pandas 3.0.6), threads and processes schedulers |
| `ray` | `integrations.ray` | `Decide(x, "text", fields=None, prefix="")` for `Dataset.map_batches`: an instance for tasks, or the class with `fn_constructor_args=(build,)` for an actor pool (built once per actor); dict-of-arrays and pandas batches | ray[data] 2.58.0 (pandas 3.0.6), real local cluster |
| `chroma` | `integrations.chroma` | `tag(x, documents, metadatas)` / `add(collection, x, ids=..., documents=..., ...)`: decisions as metadata at ingest (filter with `where=`); `filter_results(x, results, field, keep, query=...)`: drop the hits of `collection.query` / `get` whose decision is not in `keep` | chromadb 1.5.9, in-memory `EphemeralClient` |
| `qdrant` | `integrations.qdrant` | `tag(x, texts, payloads)` / `points(x, ids, vectors, texts)`: decisions in the payload at ingest (filter with a payload `Filter`); `filter_points(x, hits, field, keep, query=...)` after `query_points` / `scroll` | qdrant-client 1.19.1, `QdrantClient(":memory:")` |

Framework LLM objects are recognised by the module of their class, so `teacher=ChatOpenAI(...)` just works; the
framework is imported only when the wrapped teacher is first used.

<!-- no-test: needs the litellm extra and a LiteLLM proxy config; examples/04-integrations/litellm runs it in CI -->
```python
# guard.py next to the LiteLLM proxy config; config.yaml: guardrails: - guardrail_name: injection
#   litellm_params: {guardrail: guard.Guard, mode: pre_call}
import decisionsmith as ds
from decisionsmith.integrations.litellm import guardrail

Guard = guardrail(ds.harness(Injection, teacher="claude-haiku-4-5", student="laya"), field="is_attack", block=[True])
```

<!-- no-test: needs the openai extra; examples/04-integrations/openai runs it in CI -->
```python
from openai import OpenAI
from decisionsmith.integrations.openai import wrap

client = wrap(OpenAI(), ds.model(Ticket))  # a trained model is sure more often
r = client.chat.completions.parse(model="gpt-5-mini", messages=messages, response_format=Ticket)
r.id == "decisionsmith"  # answered locally; otherwise the API answered as usual
```

`wrap` answers only when every field of the `response_format` is a field of the model's schema (or, for
`ds.model(labels)`, a single field) and the value fits the format; anything else goes to the client unchanged.
With `DS_OFFLINE=1` the wrapped client is never called: unsure answers are used as they are.

<!-- no-test: needs the openai-agents extra; examples/04-integrations/openai-agents runs it in CI -->
```python
from agents import Agent
from decisionsmith.integrations.openai_agents import input_guardrail

guard = input_guardrail(ds.harness(Injection, teacher="claude-haiku-4-5", student="laya"), "is_attack")
agent = Agent(name="support", instructions="...", input_guardrails=[guard])
```

<!-- no-test: a fragment inside a LangChain app (prompt, llm and h come from it); examples/04-integrations/langchain runs it in CI -->
```python
from decisionsmith.integrations.langchain import DecisionRunnable, as_tool, compressor, output_parser

route = prompt | llm | output_parser(h, "team")  # or: some_runnable | DecisionRunnable(h, "team")
keep = compressor(ds.harness(Relevance, ...), "relevant", keep=[True])  # RAG relevance filter
```

LangChain's Vertex AI example uses `ChatGoogleGenerativeAI(vertexai=True)`: `langchain-google-vertexai`'s
`ChatVertexAI` is deprecated in favour of it and cannot be built offline without Google credentials. LlamaIndex is
tested with `llama-index-llms-openai` 0.7.9 because `llama-index-llms-groq` 0.6.1 requires `< 0.8`.

smolagents' `TransformersModel` example (`examples/04-integrations/smolagents/transformers/`) is not run by the
offline test: building the model loads its weights.

Structured-output libraries: Outlines ships as a teacher (above); Marvin as a `classify` drop-in. Outlines
examples cover Ollama, vLLM and OpenAI; a `transformers` example is not included because it needs model weights
that the offline example run cannot download.

The sections below are the full plan; each row moves to **Shipped** when it is built and tested.

## 1. Engines (what can answer)

| engine | string | extra | notes |
|---|---|---|---|
| Any LLM, built-in client | `"claude-sonnet-5"`, `"gpt-5"`, `"gemini-2.5-flash"`, `"groq/…"`, `"openrouter/…"`, `"ollama/qwen3"`, `ds.LLM(m, url=...)` | none (`anthropic` for Claude) | one call answers every field; one-hot answers |
| Any LiteLLM model | `"litellm/bedrock/…"`, `"litellm/vertex_ai/…"`, ... | `litellm` | cost per call from LiteLLM |
| **Jev (TypeSafe)** | `"jev"`, `"jev:<model>"` | none | hosted, `TYPESAFE_API_KEY`; teacher or zero-shot student; can't be fine-tuned, can be `adapt()`ed |
| **Laya in-process** | `"laya"`, `"laya:multilingual"`, `"laya:typed-decisions"`, `"laya:./runs/v1"`, `"laya:<org>/<repo>"` | `laya` | local student; fine-tune with `ds.finetune` / `h.finetune()` |
| **Any `/v1/systemone` endpoint** (laya-serve, laya.cpp server, other Jev-compatible servers) | `"systemone:http://host:8000"` | none | bearer token from `SYSTEMONE_API_KEY` |
| Custom engine | any object with `name` and `ask(text, questions)` (optionally `aask`) | none | returns Jev-format answers |
| Fake | `"fake"`, `ds.testing.FakeEngine(...)` | none | tests, docs, demos |

Later engine work:
| item | why |
|---|---|
| Honest LLM probabilities: log-probs over options (OpenAI-compatible, vLLM, llama.cpp, Ollama) → self-consistency → one-hot, weighted by method | better soft labels for `finetune` and a real teacher confidence |
| Per-field LLM calls when a schema has many fields (`split="field"`) | long schemas, weaker models |
| laya.cpp, laya-mlx, ONNX as named engines | only after a local parity check against the reference Laya runtime; until then use `systemone:<url>` |

## 2. LLM gateways, proxies and clients: v0.2

| integration | how decisionsmith fits | pri |
|---|---|---|
| **LiteLLM Proxy** guardrail | `pre_call` / `post_call` guardrail class: the harness decides allow / block | shipped |
| **LiteLLM Router** | harness picks the model tier (cheap vs strong) per request | shipped |
| **OpenAI-compatible clients** (OpenAI SDK, any `base_url` client) | wrap a client so a `response_format` with enum/boolean fields is answered by the harness | shipped |
| **Instructor** hook | same Pydantic model; answer from the harness first, fall back to the LLM call when unsure | shipped |
| **Portkey** | guardrail webhook | shipped |
| **simonw/llm** | `llm decide` plugin command; `llm` models as teachers | shipped |
| **Cloudflare AI Gateway / Kong AI Gateway** | routing / guardrail webhook recipe | v0.2 |
| **OpenRouter** | as a teacher engine (built in: `"openrouter/<model>"`) + routing recipe | teacher shipped |
| Vercel AI Gateway | recipe | later |

## 3. Agent frameworks: v0.2

| framework | adapter | pri |
|---|---|---|
| **LangChain** | `DecisionRunnable` (a Runnable, works with `\|`), `as_tool()`, output parser | shipped |
| **LangGraph** | `route_on(harness, field)` for conditional edges; guard node | shipped |
| **Pydantic AI** | tool; output validator; router between agents | shipped |
| **OpenAI Agents SDK** | `@input_guardrail` / `@output_guardrail` backed by a harness; handoff router | shipped |
| **Claude Agent SDK** | `PreToolUse` / `PostToolUse` hooks | shipped (an in-process SDK tool: not yet, not in this batch) |
| **LlamaIndex** | node postprocessor (relevance filter), selector (for the router query engine), `FunctionTool` | shipped |
| **CrewAI** | tool + task guardrail | shipped |
| **DSPy** | module whose forward is a harness decision; metric | shipped |
| **Haystack** | component (router, document filter) | shipped |
| **AutoGen** | `FunctionTool` + team termination condition | shipped (a message-dropping intervention handler: not yet, autogen-core 0.7.5 never marks a dropped message done, so `stop_when_idle()` hangs) |
| **Google ADK** | tool + `before_model_callback` guard | shipped |
| **smolagents** | tool | shipped |
| **Semantic Kernel** | filter + plugin function | shipped |
| **Agno** | tool | shipped |
| Mastra (TS), Vercel AI SDK (TS) | via the TS client (§ 9) | later |

<!-- no-test: a fragment inside a LangGraph app (graph and ticket_harness come from it); examples/04-integrations/langgraph runs it in CI -->
```python
from decisionsmith.integrations.langgraph import route_on

graph.add_conditional_edges(
    "intake",
    route_on(ticket_harness, "team"),
    {"billing": "billing_agent", "technical": "tech_agent", "sales": "sales_agent"},
)
```

## 4. Web backends and jobs: v0.2

| integration | adapter | pri |
|---|---|---|
| **FastAPI** | `Depends(ds.dependency(harness))`; `ds.Result[T]` as `response_model`; label endpoint | v0.2 |
| **Django** | app with a model admin for labels + middleware helper | v0.2 |
| **Flask** | extension | v0.2 |
| **Celery / RQ / Dramatiq** | batch task using `h.many()` | v0.2 |
| Temporal, Prefect, Airflow | activity / task recipes | later |

## 5. Data and RAG stores

Decide a whole table in batches (the production flow: nightly jobs, backfills, reports), or train on a dataset by
name (the developer flow). Every helper takes a `ds.model(...)` or a `ds.harness(...)` and adds, per field, the
value, `<field>_confidence` (the top probability) and `<field>_source` (`student` or `teacher`). Missing or empty
texts give empty values instead of an error. The shipped rows are in the table above; examples are in
`examples/04-integrations/<name>/`.

<!-- no-test: needs the pandas, duckdb and datasets extras; examples/04-integrations runs each in CI -->
```python
import decisionsmith as ds
import decisionsmith.integrations.pandas  # adds df.ds
from decisionsmith.integrations.duckdb import register

h = ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya")
df = df.ds.decide("text", h, prefix="pred_")  # + pred_team, pred_team_confidence, pred_team_source, ...
register(con, h)  # SELECT ds_decide(text, 'team'), ds_confidence(text, 'team') FROM tickets WHERE ds_is_true(text)
model = ds.model(Ticket)
model.train("hf:my-org/tickets")  # a hub dataset or a local folder; "hf:my-org/tickets:test" picks a split
```

Options people change: `fields=` (only some fields), `prefix=` (keep the new columns apart from your own labels),
`batch_size=` (texts per model call, default 256), and for DuckDB `field=` (which yes/no field `ds_is_true` reads).

What can go wrong:
- *A harness can't be pickled* (its SQLite log), so Spark, Ray actors and process-based Dask take a function that
  builds it: `udf(lambda: ds.harness(...), "team")`. The function runs in each worker; point `log=` at a path the
  workers can write, or `log=None`.
- *Spark needs Java 17+* and a worker Python with decisionsmith installed: set `PYSPARK_PYTHON` to it.
- *Chroma metadata can't hold nulls*: an empty document gets no decision keys (and `None` metadata when nothing is left).
- *`hf:` on Windows*: a drive path (`C:\...`) is not supported; use a relative path or a hub id. The CLI's
  `finetune` / `bench` take files only; use Python for `hf:`.

| integration | status |
|---|---|
| pandas, Polars, HF datasets, DuckDB, Spark, Dask, Ray Data, Chroma, Qdrant | shipped (above) |
| pgvector | not yet: needs a running Postgres server, which the offline tests can't provide; the pandas or DuckDB helpers tag rows before you insert them |
| BigQuery, Snowflake, Databricks | not yet: need cloud accounts |

Next: [examples](../examples/README.md) (the gallery lists every data example).

## 6. Observability and evaluation: v0.2

| integration | adapter | pri |
|---|---|---|
| **OpenTelemetry** | a span per decision: engine, field, source, confidence, mode, latency, cost | v0.2 |
| **Langfuse** | trace + score export, decisions as observations | v0.2 |
| **LangSmith** | run export + dataset sync for labels | v0.2 |
| **Arize Phoenix** | OTel semantic conventions | v0.2 |
| **Prometheus** | metrics: decisions by source, sure-rate, teacher cost | v0.2 |
| **MLflow** | log `finetune` runs, reports and checkpoints | v0.2 |
| **Weights & Biases** | log `finetune` runs and reports | v0.2 |
| **promptfoo / DeepEval** | harness as provider / scorer | v0.2 |
| pytest | `ds.testing.FakeEngine` for users' own tests | v0.1 |

## 7. Labelling (feeds `label()` and `finetune`)

| integration | pri |
|---|---|
| CLI review queue (`decisionsmith review`: uncertain, disagreement, random) | v0.2 |
| **Label Studio** ML backend (pre-annotations from the harness, labels back into the log) | v0.2 |
| **Argilla** (records + suggestions from the harness, feedback back into the log) | v0.2 |
| cleanlab (label-noise candidates in the report) | later |

## 8. Agents and CLIs

See [agents.md](agents.md): MCP for Claude Code/Desktop, Codex, Gemini CLI, Cursor, VS Code, Windsurf, Zed, Cline,
Continue, Goose, OpenCode; the Claude Code plugin with skills/commands/subagents; AGENTS.md + `--json` CLI. v0.1.

## 9. TypeScript and other languages

v0.2: `@decisionsmith/client` against a small `decisionsmith serve` (Jev-compatible `/v1/systemone` plus
`/v1/decide`), so Vercel AI SDK, Mastra and LangChain.js apps can call a harness. Meanwhile TS users can use
Laya's `laya-ts` as a student and any LLM SDK as teacher following the same pattern.

## 10. Rules for every integration

1. Thin: ≤ ~100 lines over the public API.
2. Native: the framework's own base classes and types.
3. Opt-in extra: `uv add "decisionsmith[langchain]"`; import path `decisionsmith.integrations.<name>`.
4. Tested with `FakeEngine` against a pinned framework version; one example folder each; ≥ 95% coverage.
5. New third-party *runtimes* (engines that change answers) pass a local parity check against the reference Laya
   runtime first.
6. Never send data anywhere the user didn't configure; secrets from the environment only.
