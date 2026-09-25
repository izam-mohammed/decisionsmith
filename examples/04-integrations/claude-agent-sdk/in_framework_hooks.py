"""A `PreToolUse` hook: the harness decides whether each Bash command is destructive and denies it if so
(ANTHROPIC_API_KEY; the SDK runs the Claude Code CLI).

With DS_OFFLINE=1 the hook is called directly on two sample commands instead of starting a session.
"""

import asyncio
import os

from _schema import Risky
from claude_agent_sdk import ClaudeAgentOptions, query

import decisionsmith as ds
from decisionsmith.integrations.claude_agent_sdk import hook

h = ds.harness(Risky, teacher="claude-haiku-4-5", student="laya")
bash = hook(h, field="is_destructive", block=[True], matcher="Bash")
options = ClaudeAgentOptions(allowed_tools=["Bash"], hooks={"PreToolUse": [bash]})


async def main():
    if os.environ.get("DS_OFFLINE"):
        for command in ["ls -la", "rm -rf / --no-preserve-root"]:
            data = {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": command}}
            out = await bash.hooks[0](data, None, {"signal": None})
            print("deny:" if out else "allow:", command)
        return
    async for message in query(prompt="Tidy up the build folder in this repo", options=options):
        print(message)


asyncio.run(main())
