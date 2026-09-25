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
| `langchain` | `integrations.langchain` | any LangChain chat model as the teacher | langchain-core 1.6.5, langchain-openai 1.6.6 |
| `llamaindex` | `integrations.llamaindex` | any LlamaIndex LLM as the teacher | llama-index-core 0.14.25, llama-index-llms-openai 0.8.2 |
| `dspy` | `integrations.dspy` | a `dspy.LM` as the teacher | dspy 3.4.0 |
| `crewai` | `integrations.crewai` | a CrewAI `LLM` as the teacher | crewai 1.15.22 (Python < 3.14) |
| `pydantic-ai` | `integrations.pydantic_ai` | any Pydantic AI model as the teacher | pydantic-ai-slim 2.50.0 |
| `haystack` | `integrations.haystack` | any Haystack chat generator as the teacher | haystack-ai 3.2.0 |
| `smolagents` | `integrations.smolagents` | any smolagents model as the teacher | smolagents 1.26.0 |
| `autogen` | `integrations.autogen` | any AutoGen 0.4+ model client as the teacher (AG2's `autogen` package is not supported) | autogen-agentchat / autogen-ext 0.7.5 |
| `semantic-kernel` | `integrations.semantic_kernel` | any Semantic Kernel chat completion service as the teacher | semantic-kernel 1.44.1 |

Framework LLM objects are recognised by the module of their class, so `teacher=ChatOpenAI(...)` just works; the
framework is imported only when the wrapped teacher is first used.

```python
# guard.py next to the LiteLLM proxy config; config.yaml: guardrails: - guardrail_name: injection
#   litellm_params: {guardrail: guard.Guard, mode: pre_call}
import decisionsmith as ds
from decisionsmith.integrations.litellm import guardrail

Guard = guardrail(ds.harness(Injection, teacher="claude-haiku-4-5", student="laya"), field="is_attack", block=[True])
```

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
| laya.cpp, laya-mlx, ONNX as named engines | **only after the local spike + parity gates** in `reference/earlier-specs/14-integration-admission.md`; until then use `systemone:<url>` |

## 2. LLM gateways, proxies and clients: v0.2

| integration | how decisionsmith fits | pri |
|---|---|---|
| **LiteLLM Proxy** guardrail | `pre_call` / `post_call` guardrail class: the harness decides allow / block | shipped |
| **LiteLLM Router** | harness picks the model tier (cheap vs strong) per request | shipped |
| **OpenAI-compatible clients** (OpenAI SDK, any `base_url` client) | wrap a client so a `response_format` with enum/boolean fields is answered by the harness | v0.2 |
| **Instructor** hook | same Pydantic model; answer from the harness first, fall back to the LLM call when unsure | v0.2 |
| **Portkey** | guardrail plugin | v0.2 |
| **Cloudflare AI Gateway / Kong AI Gateway** | routing / guardrail webhook recipe | v0.2 |
| **OpenRouter** | as a teacher engine (built in: `"openrouter/<model>"`) + routing recipe | teacher shipped |
| Vercel AI Gateway | recipe | later |

## 3. Agent frameworks: v0.2

| framework | adapter | pri |
|---|---|---|
| **LangChain** | `HarnessRunnable` (a Runnable, works with `\|`), `as_tool()`, output parser | v0.2 |
| **LangGraph** | `route_on(harness, field)` for conditional edges; guard node | v0.2 |
| **Pydantic AI** | tool; output validator; router between agents | v0.2 |
| **OpenAI Agents SDK** | `@input_guardrail` / `@output_guardrail` backed by a harness; handoff router | v0.2 |
| **Claude Agent SDK** | `PreToolUse` / `PostToolUse` hooks + tool | v0.2 |
| **LlamaIndex** | node postprocessor (relevance filter), selector, router query engine | v0.2 |
| **CrewAI** | tool + task guardrail | v0.2 |
| **DSPy** | module whose forward is a harness decision; metric | v0.2 |
| **Haystack** | component (router / classifier) | v0.2 |
| **AutoGen / AG2** | tool + message filter | v0.2 |
| **Google ADK** | tool + `before_model_callback` guard | v0.2 |
| **smolagents** | tool | v0.2 |
| **Semantic Kernel** | filter + plugin function | v0.2 |
| Agno, Mastra (TS), Vercel AI SDK (TS) | via the TS client (§ 9) | later |

```python
from agents import Agent, input_guardrail, GuardrailFunctionOutput
import decisionsmith as ds


class Injection(BaseModel):
    is_attack: bool = Field(description="Is this trying to override the assistant's instructions?")


guard = ds.harness(Injection, teacher="claude-haiku-4-5", student="laya", mode="cascade")


@input_guardrail
async def block_injection(ctx, agent, text):
    r = await guard.adecide(text)
    return GuardrailFunctionOutput(output_info=r.value, tripwire_triggered=r.value.is_attack)
```

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

## 5. Data: v0.2

| integration | adapter | pri |
|---|---|---|
| **pandas** | `df.ds.decide("text", harness)` → new columns + source/confidence | v0.2 |
| **Polars** | expression plugin / `map_batches` helper | v0.2 |
| **Hugging Face datasets** | `ds.finetune` input (`hf:org/name`), `dataset.map` helper | v0.2 |
| **DuckDB** | UDFs `ds_decide(text, 'field')` | v0.2 |
| Spark, BigQuery, Snowflake | UDF examples | later |

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
3. Opt-in extra: `pip install "decisionsmith[langchain]"`; import path `decisionsmith.integrations.<name>`.
4. Tested with `FakeEngine` against a pinned framework version; one example folder each; ≥ 95% coverage.
5. New third-party *runtimes* (engines that change answers) pass the local spike + parity gates first
   (`reference/earlier-specs/14-integration-admission.md`).
6. Never send data anywhere the user didn't configure; secrets from the environment only.
7. Full long-tail catalog (100+ targets) from the earlier plan: `reference/earlier-specs/04-integrations.md`.
