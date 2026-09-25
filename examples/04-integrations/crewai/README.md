# CrewAI

A CrewAI LLM as the teacher (OpenAI, Claude, Gemini, Ollama, Azure), a BaseTool backed by your model, and a task guardrail that sends blocked outputs back to the agent.

```python
"""A task guardrail: your model reads the agent's answer, and CrewAI asks again when it promises a refund
(OPENAI_API_KEY).

With DS_OFFLINE=1 the guardrail is called on two task outputs instead of kicking off the crew.
"""

import os

from _schema import Reply
from crewai import Agent, Crew, Task
from crewai.tasks.task_output import TaskOutput

import decisionsmith as ds
from decisionsmith.integrations.crewai import guardrail

h = ds.harness(Reply, teacher="gpt-5-mini", student="laya")
check = guardrail(h, "promises_refund", block=[True], message="Do not promise a refund; answer again.")

if os.environ.get("DS_OFFLINE"):
    for raw in ["Thanks, our billing team will look into the double charge.", "Yes, we will refund you today."]:
        ok, _ = check(TaskOutput(description="reply", raw=raw, agent="support"))
        print("pass:" if ok else "retry:", raw)
else:
    agent = Agent(
        role="Support agent", goal="Answer customers politely", backstory="Never promise refunds.", llm="gpt-5-mini"
    )
    task = Task(
        description="Reply to: {message}",
        expected_output="A short reply",
        agent=agent,
        guardrail=check,
        guardrail_max_retries=2,
    )
    print(Crew(agents=[agent], tasks=[task]).kickoff(inputs={"message": "I was charged twice, refund me"}).raw)
```

| file | what it shows |
|---|---|
| [`in_framework_guardrail.py`](in_framework_guardrail.py) | A task guardrail: your model reads the agent's answer, and CrewAI asks again when it promises a refund |
| [`in_framework_tool.py`](in_framework_tool.py) | A CrewAI tool: the agent calls `route_ticket` and your harness answers (OPENAI_API_KEY). |
| [`teacher_anthropic.py`](teacher_anthropic.py) | Claude through CrewAI (ANTHROPIC_API_KEY; uv add "crewai[anthropic]"). |
| [`teacher_azure.py`](teacher_azure.py) | An Azure OpenAI deployment through CrewAI (AZURE_API_KEY, AZURE_ENDPOINT). |
| [`teacher_gemini.py`](teacher_gemini.py) | Gemini through CrewAI (GEMINI_API_KEY; uv add "crewai[google-genai]"). |
| [`teacher_ollama.py`](teacher_ollama.py) | A local Ollama model through CrewAI (ollama pull qwen3). |
| [`teacher_openai.py`](teacher_openai.py) | OpenAI through CrewAI (OPENAI_API_KEY). |

## Run

```bash
uv add "decisionsmith[crewai,laya]"
uv add crewai[anthropic,google-genai,azure-ai-inference]
export AZURE_API_KEY=...
export AZURE_ENDPOINT=...
uv run python examples/04-integrations/crewai/in_framework_guardrail.py
uv run python examples/04-integrations/crewai/in_framework_tool.py
uv run python examples/04-integrations/crewai/teacher_anthropic.py
uv run python examples/04-integrations/crewai/teacher_azure.py
uv run python examples/04-integrations/crewai/teacher_gemini.py
uv run python examples/04-integrations/crewai/teacher_ollama.py
uv run python examples/04-integrations/crewai/teacher_openai.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
