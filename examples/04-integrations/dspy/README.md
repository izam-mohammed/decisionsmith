# DSPy

Any dspy.LM as the teacher (OpenAI, Claude, Gemini, Ollama, any LiteLLM id), a dspy.Module whose forward is a decision, and a metric for dspy.Evaluate.

```python
"""Score a program with `dspy.Evaluate` and a decisionsmith metric (per-field match with the gold labels).

The devset is synthetic; use your own labelled examples. The same metric works for any DSPy program that
outputs `team`, e.g. dspy.Predict("text -> team") on a big LLM, so both can be compared on one devset.
"""

import dspy
from _schema import Ticket

import decisionsmith as ds
from decisionsmith.integrations.dspy import DecisionModule, metric

GOLD = [
    ("I was charged twice this month, please refund me", "billing", True),
    ("The app crashes every time I log in", "technical", False),
    ("Can I get a quote for 20 seats?", "sales", False),
    ("My invoice shows the wrong company name", "billing", False),
]
devset = [dspy.Example(text=t, team=team, wants_refund=r).with_inputs("text") for t, team, r in GOLD]

h = ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya")
evaluate = dspy.Evaluate(devset=devset, metric=metric("team"), num_threads=1)
print("team match: %.0f%%" % evaluate(DecisionModule(h)).score)
print("all fields: %.0f%%" % dspy.Evaluate(devset=devset, metric=metric(), num_threads=1)(DecisionModule(h)).score)
```

| file | what it shows |
|---|---|
| [`in_framework_metric.py`](in_framework_metric.py) | Score a program with `dspy.Evaluate` and a decisionsmith metric (per-field match with the gold labels). |
| [`in_framework_module.py`](in_framework_module.py) | A `dspy.Module` backed by the harness: use it wherever a DSPy program goes, or inside a bigger one. |
| [`teacher_anthropic.py`](teacher_anthropic.py) | Claude through DSPy (ANTHROPIC_API_KEY). |
| [`teacher_any_litellm_id.py`](teacher_any_litellm_id.py) | Any LiteLLM model id through DSPy, here Llama 3.3 on Groq (GROQ_API_KEY). |
| [`teacher_gemini.py`](teacher_gemini.py) | Gemini through DSPy (GEMINI_API_KEY). |
| [`teacher_ollama.py`](teacher_ollama.py) | A local Ollama model through DSPy (ollama pull qwen3). |
| [`teacher_openai.py`](teacher_openai.py) | OpenAI through DSPy (OPENAI_API_KEY). |

## Run

```bash
uv add "decisionsmith[dspy,laya]"
uv run python examples/04-integrations/dspy/in_framework_metric.py
uv run python examples/04-integrations/dspy/in_framework_module.py
uv run python examples/04-integrations/dspy/teacher_anthropic.py
uv run python examples/04-integrations/dspy/teacher_any_litellm_id.py
uv run python examples/04-integrations/dspy/teacher_gemini.py
uv run python examples/04-integrations/dspy/teacher_ollama.py
uv run python examples/04-integrations/dspy/teacher_openai.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
