# Teachers: any LLM, no LLM library

The teacher is the LLM that answers while the student (Laya, or Jev) learns. decisionsmith talks to LLMs itself:
the core needs only `pydantic` and `httpx`. Give the teacher as a string, as `ds.LLM(...)`, or as the LLM object of
the framework you already use.

```python
h = ds.harness(Ticket, teacher="gpt-5-mini", student="laya")
model.train(texts, teacher="claude-haiku-4-5")
model.train(generate=300, teacher=ds.LLM("my-model", url="http://localhost:8000/v1"), about="support emails")
```

## Built-in providers

| string | provider | key (environment) | base URL |
|---|---|---|---|
| `gpt-*`, `o1*`, `o3*`, `o4*`, `openai/<model>` | OpenAI | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| `claude-*`, `anthropic/<model>` | Anthropic, via the official `anthropic` SDK (`pip install "decisionsmith[anthropic]"`) | `ANTHROPIC_API_KEY` | SDK default |
| `gemini-*`, `gemini/<model>` | Google Gemini (OpenAI-compatible endpoint) | `GEMINI_API_KEY` | `https://generativelanguage.googleapis.com/v1beta/openai` |
| `groq/<model>` | Groq | `GROQ_API_KEY` | `https://api.groq.com/openai/v1` |
| `openrouter/<model>` | OpenRouter | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` |
| `together/<model>` | Together AI | `TOGETHER_API_KEY` | `https://api.together.ai/v1` |
| `fireworks/<model>` | Fireworks AI | `FIREWORKS_API_KEY` | `https://api.fireworks.ai/inference/v1` |
| `deepseek/<model>` | DeepSeek | `DEEPSEEK_API_KEY` | `https://api.deepseek.com` |
| `xai/<model>` | xAI | `XAI_API_KEY` | `https://api.x.ai/v1` |
| `mistral/<model>` | Mistral | `MISTRAL_API_KEY` | `https://api.mistral.ai/v1` |
| `ollama/<model>` | Ollama on this machine | none | `http://localhost:11434/v1` |
| `ds.LLM(model, url=..., api_key=...)` | any OpenAI-compatible server (vLLM, LM Studio, llama.cpp server, SGLang, a gateway) | `api_key=` | `url` |
| `litellm/<any LiteLLM id>` | everything LiteLLM supports (Bedrock, Vertex, Azure, Cohere, ...) (`pip install "decisionsmith[litellm]"`) | the provider's | LiteLLM's |

Base URLs were checked against each provider's documentation on 25/09/2026.

`ds.LLM(model, url=None, api_key=None, timeout=60, retries=2, **options)`: `options` go into every request
(`max_tokens=200`, `reasoning_effort="low"`, ...) and win over decisionsmith's defaults.

## How a teacher answers

- One call answers every field. The prompt is built from the schema; the text is fenced as data.
- OpenAI-compatible servers get `response_format` with a strict JSON Schema. On a 400 that says an option is not
  supported, the client drops it and tries again: `temperature` first (reasoning models only take the default), then
  `json_schema` becomes `json_object`, then a plain prompt. The reply is parsed (code fences are fine) and validated;
  an invalid reply gets one retry with the validation error.
- Claude uses the SDK's `messages.parse(..., output_format=<answers model>)`; a `refusal` stop reason raises
  `ds.EngineError`. The SDK's own retries apply.
- Timeouts, 408, 429 and 5xx are retried with backoff (`Retry-After` is honoured, capped at 20 s).
- Answers are one-hot. Usage tokens are recorded in the log; `cost_usd` is only known through the
  LiteLLM teacher, so `bench` shows no cost for the built-in client.

## Framework LLM objects as the teacher

Pass the object; decisionsmith recognises it from its class and wraps it (the framework is never imported by
`import decisionsmith`). Install the matching extra.

| framework | objects | extra |
|---|---|---|
| LangChain | any chat model: `ChatOpenAI`, `ChatAnthropic`, `ChatOllama`, `ChatGoogleGenerativeAI`, ... | `langchain` |
| LlamaIndex | any LLM: `llama_index.llms.openai.OpenAI`, `Anthropic`, `Ollama`, ... | `llamaindex` |
| DSPy | `dspy.LM(...)` | `dspy` |
| CrewAI | `crewai.LLM(...)` | `crewai` |
| Pydantic AI | any model: `OpenAIChatModel`, `AnthropicModel`, `GoogleModel`, `TestModel`, ... | `pydantic-ai` |
| Haystack | chat generators: `OpenAIChatGenerator`, ... (a custom component: `integrations.haystack.teacher(obj)`) | `haystack` |
| smolagents | `OpenAIServerModel`, `LiteLLMModel`, `TransformersModel`, `InferenceClientModel`, ... | `smolagents` |
| AutoGen (0.4+) | model clients: `OpenAIChatCompletionClient`, ... | `autogen` |
| Semantic Kernel | chat completion services: `OpenAIChatCompletion`, ... | `semantic-kernel` |
| Agno | any model: `OpenAIChat`, `Claude`, `Gemini`, `Ollama`, ... | `agno` |
| Outlines | any Outlines model: `outlines.from_transformers(...)`, `from_ollama`, `from_vllm`, `from_openai`, ... (the answer schema is enforced) | `outlines` |
| simonw/llm | any `llm` model: `llm.get_model("gpt-5-mini")`, plugin models, async models | `llm-plugin` |
| OpenAI SDK | a client plus a model id: `integrations.openai.teacher(OpenAI(base_url=...), "model")` | `openai` |

```python
from langchain_anthropic import ChatAnthropic

h = ds.harness(Ticket, teacher=ChatAnthropic(model="claude-haiku-4-5"), student="laya")
```

Anything else with a `complete(system, user) -> str` shape becomes a teacher through
`decisionsmith.engines.TextEngine(name, complete)`, or write an engine (`name` + `ask(text, questions)`).

## Async

`await h.adecide(text)`, `await h.acall(text)` and `await model.apredict(text_or_list)`. HTTP engines (every LLM
teacher, Jev, `systemone:` URLs) run natively async; Laya and custom engines run in a thread. Routing, fallbacks and
logging are the same code as the sync path.

## Offline mode

`DS_OFFLINE=1` makes every LLM teacher, Jev and `systemone:` engine answer locally with a keyword stand-in (a label
named in the text wins, otherwise a stable hash picks one), and `DS_LAYA=<dir>` swaps the named Laya checkpoints for a
local one. Nothing leaves the machine. It is how CI runs every example and notebook; it is also a way to walk through
a flow before you have keys. Offline answers are not real labels: never train a model you mean to use on them.
