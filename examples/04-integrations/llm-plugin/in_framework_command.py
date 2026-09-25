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
