# Outlines

Any Outlines model (Ollama, vLLM, OpenAI, transformers, llama.cpp, ...) as the teacher, with structured generation.

```python
"""A local Ollama model as the teacher, with Outlines structured generation (ollama pull qwen3).

Outlines constrains the reply to the answer schema, so small local models always return valid labels.
"""

import ollama
import outlines
from _schema import TEXTS, Ticket

import decisionsmith as ds

teacher = outlines.from_ollama(ollama.Client(), "qwen3")
h = ds.harness(Ticket, teacher=teacher, student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
```

| file | what it shows |
|---|---|
| [`teacher_ollama.py`](teacher_ollama.py) | A local Ollama model as the teacher, with Outlines structured generation (ollama pull qwen3). |
| [`teacher_openai.py`](teacher_openai.py) | An OpenAI model through Outlines (OPENAI_API_KEY): Outlines sends the answer schema as structured output. |
| [`teacher_vllm.py`](teacher_vllm.py) | A vLLM server as the teacher through Outlines (vllm serve Qwen/Qwen3-8B); vLLM enforces the answer schema. |

## Run

```bash
pip install "decisionsmith[outlines,laya]"
pip install ollama
python examples/04-integrations/outlines/teacher_ollama.py
python examples/04-integrations/outlines/teacher_openai.py
python examples/04-integrations/outlines/teacher_vllm.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
