# Pydantic AI

Any Pydantic AI model as the teacher (OpenAI, Claude, Gemini, Groq, Mistral, Bedrock, Ollama), and a tool, an output validator and an agent router backed by your model.

```python
"""An output validator: your model reads every reply, and the agent answers again when it promises a refund.

Uses OPENAI_API_KEY; offline (DS_OFFLINE=1) the agent runs on Pydantic AI's TestModel with a fixed reply.
"""

import os

from pydantic import BaseModel, Field
from pydantic_ai import Agent, UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel

import decisionsmith as ds
from decisionsmith.integrations.pydantic_ai import output_validator


class Reply(BaseModel):
    promises_refund: bool = Field(description="Does this reply promise the customer a refund?")


check = ds.harness(Reply, teacher="gpt-5-mini", student="laya")
llm = TestModel(custom_output_text="Thanks, our team will look into it.") if os.environ.get("DS_OFFLINE") else None
agent = Agent(llm or "openai:gpt-5-mini", instructions="You are a support agent. Never promise refunds.", retries=2)
agent.output_validator(output_validator(check, "promises_refund", message="Do not promise a refund; answer again."))

try:
    print(agent.run_sync("I was charged twice, I want my money back").output)
except UnexpectedModelBehavior:
    print("every reply promised a refund; escalate to a human")
```

| file | what it shows |
|---|---|
| [`in_framework_output_validator.py`](in_framework_output_validator.py) | An output validator: your model reads every reply, and the agent answers again when it promises a refund. |
| [`in_framework_router.py`](in_framework_router.py) | Route each ticket to a team agent with your model instead of a triage LLM call (OPENAI_API_KEY). |
| [`in_framework_tool.py`](in_framework_tool.py) | A Pydantic AI agent that calls your model as a tool to route tickets (OPENAI_API_KEY). |
| [`teacher_anthropic.py`](teacher_anthropic.py) | Claude through Pydantic AI (ANTHROPIC_API_KEY). |
| [`teacher_bedrock.py`](teacher_bedrock.py) | Amazon Nova on Bedrock through Pydantic AI (AWS credentials with Bedrock access). |
| [`teacher_gemini.py`](teacher_gemini.py) | Gemini through Pydantic AI (GOOGLE_API_KEY). |
| [`teacher_groq.py`](teacher_groq.py) | Groq through Pydantic AI (GROQ_API_KEY). |
| [`teacher_mistral.py`](teacher_mistral.py) | Mistral through Pydantic AI (MISTRAL_API_KEY). |
| [`teacher_ollama.py`](teacher_ollama.py) | A local Ollama model through Pydantic AI (ollama pull qwen3); no key, nothing leaves the machine. |
| [`teacher_openai.py`](teacher_openai.py) | OpenAI through Pydantic AI (OPENAI_API_KEY). |

## Run

```bash
pip install "decisionsmith[pydantic-ai,laya]"
pip install pydantic-ai-slim[openai,anthropic,google,groq,mistral,bedrock]
python examples/04-integrations/pydantic-ai/in_framework_output_validator.py
python examples/04-integrations/pydantic-ai/in_framework_router.py
python examples/04-integrations/pydantic-ai/in_framework_tool.py
python examples/04-integrations/pydantic-ai/teacher_anthropic.py
python examples/04-integrations/pydantic-ai/teacher_bedrock.py
python examples/04-integrations/pydantic-ai/teacher_gemini.py
python examples/04-integrations/pydantic-ai/teacher_groq.py
python examples/04-integrations/pydantic-ai/teacher_mistral.py
python examples/04-integrations/pydantic-ai/teacher_ollama.py
python examples/04-integrations/pydantic-ai/teacher_openai.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
