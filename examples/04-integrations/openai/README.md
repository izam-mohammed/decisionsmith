# OpenAI SDK

Any OpenAI SDK client (OpenAI, Claude, Gemini, Groq, Ollama via base_url) as the teacher, and wrap(client) so your model answers structured calls first.

```python
"""Wrap an OpenAI client: requests whose `response_format` only asks for decisions are answered by your model
when it is sure; everything else, and anything it is unsure about, goes to the API as before (OPENAI_API_KEY).

Train the model first (`model.train(...)`) so it is sure more often; base Laya mostly falls through.
"""

from _schema import TEXTS, Ticket
from openai import OpenAI

import decisionsmith as ds
from decisionsmith.integrations.openai import wrap

client = wrap(OpenAI(), ds.model(Ticket))

for text in TEXTS[:3]:
    r = client.chat.completions.parse(
        model="gpt-5-mini",
        messages=[{"role": "system", "content": "Route the support ticket."}, {"role": "user", "content": text}],
        response_format=Ticket,
    )
    who = "decisionsmith" if r.id == "decisionsmith" else "the API"
    print(r.choices[0].message.parsed, "<- answered by", who)
```

| file | what it shows |
|---|---|
| [`in_framework_wrap.py`](in_framework_wrap.py) | Wrap an OpenAI client: requests whose `response_format` only asks for decisions are answered by your model |
| [`teacher_anthropic.py`](teacher_anthropic.py) | Claude through Anthropic's OpenAI SDK compatibility endpoint (ANTHROPIC_API_KEY). |
| [`teacher_gemini.py`](teacher_gemini.py) | Gemini through its OpenAI-compatible endpoint (GEMINI_API_KEY). |
| [`teacher_groq.py`](teacher_groq.py) | Groq through the OpenAI SDK (GROQ_API_KEY). |
| [`teacher_ollama.py`](teacher_ollama.py) | A local Ollama model through the OpenAI SDK (ollama pull qwen3); no key, nothing leaves the machine. |
| [`teacher_openai.py`](teacher_openai.py) | OpenAI through the official SDK (OPENAI_API_KEY). |

## Run

```bash
uv add "decisionsmith[openai,laya]"
uv run python examples/04-integrations/openai/in_framework_wrap.py
uv run python examples/04-integrations/openai/teacher_anthropic.py
uv run python examples/04-integrations/openai/teacher_gemini.py
uv run python examples/04-integrations/openai/teacher_groq.py
uv run python examples/04-integrations/openai/teacher_ollama.py
uv run python examples/04-integrations/openai/teacher_openai.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
