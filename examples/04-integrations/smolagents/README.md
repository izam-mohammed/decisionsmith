# smolagents

A decisionsmith Tool for smolagents CodeAgent and ToolCallingAgent, and any smolagents model (OpenAI-compatible server, LiteLLM, Hugging Face Inference Providers) as the teacher.

```python
"""A smolagents `CodeAgent` with a `route_ticket` tool: the agent writes code that calls it and the harness answers
(OPENAI_API_KEY).

With DS_OFFLINE=1 the tool is called directly instead of running the agent's LLM.
"""

import os

from _schema import TEXTS, Ticket
from smolagents import CodeAgent, OpenAIServerModel

import decisionsmith as ds
from decisionsmith.integrations.smolagents import tool

h = ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya")
route = tool(h, name="route_ticket", description="Which team handles a support ticket, and is it a refund?")
agent = CodeAgent(tools=[route], model=OpenAIServerModel(model_id="gpt-5-mini"), max_steps=3)

print(route.to_code_prompt().splitlines()[0])
for text in TEXTS[:3]:
    if os.environ.get("DS_OFFLINE"):
        print(route(text=text), "<-", text)
    else:
        print(agent.run("Route this ticket with route_ticket and give the team: %s" % text))
```

| file | what it shows |
|---|---|
| [`in_framework_tool.py`](in_framework_tool.py) | A smolagents `CodeAgent` with a `route_ticket` tool: the agent writes code that calls it and the harness answers |
| [`teacher_inference_client.py`](teacher_inference_client.py) | An open model on Hugging Face Inference Providers through smolagents' InferenceClientModel (HF_TOKEN). |
| [`teacher_litellm.py`](teacher_litellm.py) | Claude through smolagents' LiteLLMModel, or any LiteLLM model id (ANTHROPIC_API_KEY). |
| [`teacher_openai_server.py`](teacher_openai_server.py) | OpenAI (or any OpenAI-compatible server: pass api_base=) through smolagents' OpenAIServerModel (OPENAI_API_KEY). |

## Run

```bash
uv add "decisionsmith[smolagents,laya]"
uv add smolagents[openai,litellm]
export HF_TOKEN=...
uv run python examples/04-integrations/smolagents/in_framework_tool.py
uv run python examples/04-integrations/smolagents/teacher_inference_client.py
uv run python examples/04-integrations/smolagents/teacher_litellm.py
uv run python examples/04-integrations/smolagents/teacher_openai_server.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
