# simonw/llm plugin

An `llm decide` command for the llm CLI, and any llm model as the teacher.

```python
"""The `llm decide` command: installing decisionsmith[llm-plugin] adds it to simonw's `llm` CLI.

    llm decide "I was charged twice" -l billing -l technical -l sales
    llm decide "..." -l billing -l technical -l sales --engine laya:./runs/v1 --teacher gpt-5-mini
    cat ticket.txt | llm decide -l billing -l technical -l sales

This script runs the same command in-process.
"""

from click.testing import CliRunner
from llm.cli import cli

labels = ["-l", "billing", "-l", "technical", "-l", "sales"]
for text in ["I was charged twice this month", "The app crashes every time I log in"]:
    out = CliRunner().invoke(cli, ["decide", text, *labels, "--question", "Which team should handle this?"])
    if out.exit_code:
        raise SystemExit(out.output)
    print(out.output.strip(), "<-", text)
```

| file | what it shows |
|---|---|
| [`in_framework_command.py`](in_framework_command.py) | The `llm decide` command: installing decisionsmith[llm-plugin] adds it to simonw's `llm` CLI. |
| [`teacher_openai.py`](teacher_openai.py) | Any `llm` model as the teacher: keys and plugins come from your llm setup (`llm keys set openai`). |

## Run

```bash
pip install "decisionsmith[llm-plugin,laya]"
python examples/04-integrations/llm-plugin/in_framework_command.py
python examples/04-integrations/llm-plugin/teacher_openai.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
