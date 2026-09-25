# Integrations

> **TL;DR**: decisionsmith plugs in two ways. As **engines** (what answers: any LLM, Jev, Laya, any systemone URL),
> which ship in v0.1. And as a **harness inside your framework** (LiteLLM, LangChain, LangGraph, Pydantic AI,
> OpenAI Agents, FastAPI, pandas, OpenTelemetry and ~30 more), planned for v0.2. Every framework integration is a
> thin adapter (≤ ~100 lines) over the public API, behind a pip extra, with its own example and a test against a
> pinned framework version. Snippets below are target designs: verify against the pinned version when building.

Priority: **v0.1** ships now · **v0.2** first month after launch · **later** on demand or by the community.

## 1. Engines (what can answer): v0.1

| engine | string | extra | notes |
|---|---|---|---|
| Any LLM via LiteLLM (Anthropic, OpenAI, Google, Bedrock, Azure, Vertex, Mistral, Groq, OpenRouter, Together, Fireworks…) | `"claude-sonnet-5"`, `"gpt-5"`, `"openrouter/…"` | `[llm]` | one call answers every field; one-hot answers |
| Local LLMs (Ollama, LM Studio, vLLM, llama.cpp server, SGLang) | `"ollama/qwen3"`, `"openai/<model>"` + `api_base` | `[llm]` | fully offline teacher |
| **Jev (TypeSafe)** | `"jev"`, `"jev:<model>"` | none | hosted, `TYPESAFE_API_KEY`; teacher or zero-shot student; can't be fine-tuned, can be `adapt()`ed |
| **Laya in-process** | `"laya"`, `"laya:multilingual"`, `"laya:typed-decisions"`, `"laya:./runs/v1"`, `"laya:<org>/<repo>"` | `[laya]` | local student; fine-tune with `ds.finetune` / `h.finetune()` |
| **Any `/v1/systemone` endpoint** (laya-serve, laya.cpp server, other Jev-compatible servers) | `"systemone:http://host:8000"` | none | bearer token from `SYSTEMONE_API_KEY` |
| Custom engine | any object with `name` and `ask(text, questions)` | none | returns Jev-format answers |
| Fake | `"fake"`, `ds.testing.FakeEngine(...)` | none | tests, docs, demos |

v0.2 engine work:
| item | why |
|---|---|
| Honest LLM probabilities: log-probs over options (OpenAI-compatible, vLLM, llama.cpp, Ollama) → self-consistency → one-hot, weighted by method | better soft labels for `finetune` and a real teacher confidence |
| Per-field LLM calls when a schema has many fields (`split="field"`) | long schemas, weaker models |
| Async engines (`ask_async`) + `h.adecide()` | FastAPI, agents, guardrails need async |
| laya.cpp, laya-mlx, ONNX as named engines | **only after the local spike + parity gates** in `reference/earlier-specs/14-integration-admission.md`; until then use `systemone:<url>` |

## 2. LLM gateways, proxies and clients: v0.2

| integration | how decisionsmith fits | pri |
|---|---|---|
| **LiteLLM Proxy** guardrail | `pre_call` / `post_call` guardrail class: the harness decides allow / block / flag | v0.2 |
| **LiteLLM Router** | harness picks the model tier (cheap vs strong) per request | v0.2 |
| **OpenAI-compatible clients** (OpenAI SDK, any `base_url` client) | wrap a client so a `response_format` with enum/boolean fields is answered by the harness | v0.2 |
| **Instructor** hook | same Pydantic model; answer from the harness first, fall back to the LLM call when unsure | v0.2 |
| **Portkey** | guardrail plugin | v0.2 |
| **Cloudflare AI Gateway / Kong AI Gateway** | routing / guardrail webhook recipe | v0.2 |
| **OpenRouter** | as a teacher engine (already works via LiteLLM) + routing recipe | v0.2 |
| Vercel AI Gateway | recipe | later |

```yaml
# LiteLLM proxy config (v0.2 target)
guardrails:
  - guardrail_name: decisionsmith-injection
    litellm_params:
      guardrail: decisionsmith.integrations.litellm.HarnessGuard
      mode: pre_call
```

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
    r = await guard.adecide(text)  # v0.2: async API
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
