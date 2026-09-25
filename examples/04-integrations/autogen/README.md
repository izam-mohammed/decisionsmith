# AutoGen

A decision as an AutoGen FunctionTool, a termination condition that stops a team when a message is blocked, and any AutoGen model client (OpenAI, Claude, Azure OpenAI, Ollama) as the teacher.

```python
"""A team that stops as soon as an agent's message leaks private data: the harness checks each message from the
agents named in `sources` (OPENAI_API_KEY).

stop_on(...) is a TerminationCondition, so it combines with the others: stop_on(...) | MaxMessageTermination(8).
Offline (DS_OFFLINE=1) replay clients play the agents' models.
"""

import asyncio

from _offline import client
from _schema import Leak
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat

import decisionsmith as ds
from decisionsmith.integrations.autogen import stop_on

h = ds.harness(Leak, teacher="claude-haiku-4-5", student="laya")
guard = stop_on(h, field="leaks_data", block=[True], sources=["writer", "editor"])


async def main():
    writer = AssistantAgent(
        "writer", model_client=client(["A draft with no account details.", "Sure, yes: the password is hunter2."])
    )
    editor = AssistantAgent("editor", model_client=client(["No changes needed, add the steps."]))
    team = RoundRobinGroupChat([writer, editor], termination_condition=guard | MaxMessageTermination(5))
    result = await team.run(task="Draft a reply to a customer asking about their account.")
    for m in result.messages:
        print("%s: %s" % (m.source, m.to_text()))
    print("stopped:", result.stop_reason)


asyncio.run(main())
```

| file | what it shows |
|---|---|
| [`in_framework_stop_on.py`](in_framework_stop_on.py) | A team that stops as soon as an agent's message leaks private data: the harness checks each message from the |
| [`in_framework_tool.py`](in_framework_tool.py) | An AssistantAgent with a `route_ticket` FunctionTool: the agent calls it and the harness answers |
| [`teacher_anthropic.py`](teacher_anthropic.py) | Claude through AutoGen (ANTHROPIC_API_KEY; uv add "autogen-ext[anthropic]"). |
| [`teacher_azure_openai.py`](teacher_azure_openai.py) | Azure OpenAI through AutoGen (AZURE_OPENAI_API_KEY). |
| [`teacher_ollama.py`](teacher_ollama.py) | A local Ollama model through AutoGen (ollama pull qwen3; uv add "autogen-ext[ollama]"). |
| [`teacher_openai.py`](teacher_openai.py) | OpenAI through AutoGen (OPENAI_API_KEY). |

## Run

```bash
uv add "decisionsmith[autogen,laya]"
uv add autogen-agentchat autogen-ext[openai,anthropic,ollama]
export AZURE_OPENAI_API_KEY=...
uv run python examples/04-integrations/autogen/in_framework_stop_on.py
uv run python examples/04-integrations/autogen/in_framework_tool.py
uv run python examples/04-integrations/autogen/teacher_anthropic.py
uv run python examples/04-integrations/autogen/teacher_azure_openai.py
uv run python examples/04-integrations/autogen/teacher_ollama.py
uv run python examples/04-integrations/autogen/teacher_openai.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
