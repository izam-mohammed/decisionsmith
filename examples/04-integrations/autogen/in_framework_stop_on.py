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
