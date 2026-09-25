# smolagents with local transformers

A local Hugging Face model loaded by smolagents' TransformersModel as the teacher.

```python
"""A local Hugging Face model through smolagents' TransformersModel (downloads Qwen/Qwen3-0.6B on first run)."""

from _schema import TEXTS, Ticket
from smolagents import TransformersModel

import decisionsmith as ds

llm = TransformersModel(model_id="Qwen/Qwen3-0.6B", max_new_tokens=256)
h = ds.harness(Ticket, teacher=llm, student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
```

## Run

```bash
pip install "decisionsmith[smolagents,laya]"
pip install smolagents[transformers]
python examples/04-integrations/smolagents/transformers/teacher_transformers.py
```

Needs the Qwen/Qwen3-0.6B weights (TransformersModel loads them when it is built, so the offline test can't run it) to run for real.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
