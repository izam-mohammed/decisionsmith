# Agno

A decision as an Agno tool, and any Agno model (OpenAI, Claude, Gemini, Ollama) as the teacher.

```python
"""An Agno agent with a `route_ticket` tool: the agent calls it and the harness answers (OPENAI_API_KEY).

With DS_OFFLINE=1 the tool is called directly instead of running the agent.
"""

import os

from _schema import TEXTS, Ticket
from agno.agent import Agent
from agno.models.openai import OpenAIChat

import decisionsmith as ds
from decisionsmith.integrations.agno import tool

h = ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya")
route = tool(h, name="route_ticket", description="Which team should handle this support ticket, and is it a refund?")
agent = Agent(model=OpenAIChat(id="gpt-5-mini"), tools=[route], instructions="Route each ticket with route_ticket.")

for text in TEXTS[:3]:
    if os.environ.get("DS_OFFLINE"):
        print(route.entrypoint(text=text), "<-", text)
    else:
        print(agent.run(text).content)
```

| file | what it shows |
|---|---|
| [`in_framework_tool.py`](in_framework_tool.py) | An Agno agent with a `route_ticket` tool: the agent calls it and the harness answers (OPENAI_API_KEY). |
| [`teacher_anthropic.py`](teacher_anthropic.py) | Claude through Agno (ANTHROPIC_API_KEY; uv add anthropic). |
| [`teacher_gemini.py`](teacher_gemini.py) | Gemini through Agno (GOOGLE_API_KEY; uv add google-genai). |
| [`teacher_ollama.py`](teacher_ollama.py) | A local Ollama model through Agno (ollama pull qwen3; uv add ollama). |
| [`teacher_openai.py`](teacher_openai.py) | OpenAI through Agno (OPENAI_API_KEY). |

## Run

```bash
uv add "decisionsmith[agno,laya]"
uv add openai anthropic google-genai ollama
uv run python examples/04-integrations/agno/in_framework_tool.py
uv run python examples/04-integrations/agno/teacher_anthropic.py
uv run python examples/04-integrations/agno/teacher_gemini.py
uv run python examples/04-integrations/agno/teacher_ollama.py
uv run python examples/04-integrations/agno/teacher_openai.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
