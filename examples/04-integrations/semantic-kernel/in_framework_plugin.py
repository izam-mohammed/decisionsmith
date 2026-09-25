"""A Semantic Kernel plugin: `tickets-route_ticket` is a kernel function the harness answers.

With auto function calling the LLM calls it: OpenAIChatPromptExecutionSettings(
function_choice_behavior=FunctionChoiceBehavior.Auto()). Here it is invoked directly.
"""

import asyncio

from _schema import TEXTS, Ticket
from semantic_kernel import Kernel

import decisionsmith as ds
from decisionsmith.integrations.semantic_kernel import plugin

h = ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya")
kernel = Kernel()
kernel.add_plugin(plugin(h, name="route_ticket", description="Which team handles a support ticket?"), "tickets")


async def main():
    for text in TEXTS:
        result = await kernel.invoke(plugin_name="tickets", function_name="route_ticket", text=text)
        print(result, "<-", text)


asyncio.run(main())
