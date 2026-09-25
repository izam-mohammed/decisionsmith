# Built-in LLM teachers

Every provider the built-in client speaks (OpenAI, Claude, Gemini, Groq, OpenRouter, Together, Fireworks, DeepSeek, xAI, Mistral, Ollama, vLLM, LM Studio, any URL) labelling for Laya.

```python
"""Claude as the teacher, through the official SDK (ANTHROPIC_API_KEY; uv add "decisionsmith[anthropic]").

The teacher labels the texts; the rows train Laya (`model.train(rows)`) or go to review first.
"""

from _schema import TEXTS, Ticket

import decisionsmith as ds

model = ds.model(Ticket)
rows = model.label(TEXTS, teacher="claude-haiku-4-5")
for row in rows:
    print(row["text"], "->", {k: max(v, key=v.get) for k, v in row["answers"].items()})
```

| file | what it shows |
|---|---|
| [`teacher_anthropic.py`](teacher_anthropic.py) | Claude as the teacher, through the official SDK (ANTHROPIC_API_KEY; uv add "decisionsmith[anthropic]"). |
| [`teacher_any_openai_compatible_url.py`](teacher_any_openai_compatible_url.py) | Any OpenAI-compatible server as the teacher (llama.cpp, SGLang, a gateway, ...): set LLM_URL and LLM_API_KEY. |
| [`teacher_deepseek.py`](teacher_deepseek.py) | DeepSeek as the teacher (DEEPSEEK_API_KEY). |
| [`teacher_fireworks.py`](teacher_fireworks.py) | Fireworks AI as the teacher (FIREWORKS_API_KEY). |
| [`teacher_gemini.py`](teacher_gemini.py) | Gemini as the teacher, through its OpenAI-compatible endpoint (GEMINI_API_KEY). |
| [`teacher_groq.py`](teacher_groq.py) | Llama 3.3 70B on Groq as the teacher (GROQ_API_KEY). |
| [`teacher_lm_studio.py`](teacher_lm_studio.py) | LM Studio as the teacher: load a model, start the local server. |
| [`teacher_mistral.py`](teacher_mistral.py) | Mistral as the teacher (MISTRAL_API_KEY). |
| [`teacher_ollama.py`](teacher_ollama.py) | A local Ollama model as the teacher: no key, nothing leaves the machine (ollama pull qwen3). |
| [`teacher_openai.py`](teacher_openai.py) | OpenAI as the teacher (OPENAI_API_KEY). |
| [`teacher_openrouter.py`](teacher_openrouter.py) | Any OpenRouter model as the teacher (OPENROUTER_API_KEY). |
| [`teacher_together.py`](teacher_together.py) | Together AI as the teacher (TOGETHER_API_KEY). |
| [`teacher_vllm.py`](teacher_vllm.py) | A vLLM server as the teacher (vllm serve Qwen/Qwen3-8B). |
| [`teacher_xai.py`](teacher_xai.py) | Grok from xAI as the teacher (XAI_API_KEY). |

## Run

```bash
uv add "decisionsmith[laya]"
uv run python examples/04-integrations/builtin/teacher_anthropic.py
uv run python examples/04-integrations/builtin/teacher_any_openai_compatible_url.py
uv run python examples/04-integrations/builtin/teacher_deepseek.py
uv run python examples/04-integrations/builtin/teacher_fireworks.py
uv run python examples/04-integrations/builtin/teacher_gemini.py
uv run python examples/04-integrations/builtin/teacher_groq.py
uv run python examples/04-integrations/builtin/teacher_lm_studio.py
uv run python examples/04-integrations/builtin/teacher_mistral.py
uv run python examples/04-integrations/builtin/teacher_ollama.py
uv run python examples/04-integrations/builtin/teacher_openai.py
uv run python examples/04-integrations/builtin/teacher_openrouter.py
uv run python examples/04-integrations/builtin/teacher_together.py
uv run python examples/04-integrations/builtin/teacher_vllm.py
uv run python examples/04-integrations/builtin/teacher_xai.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
