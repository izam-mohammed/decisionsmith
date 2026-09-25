"""An ADK agent with a `route_ticket` tool: Gemini calls it and the harness answers (GEMINI_API_KEY).

With DS_OFFLINE=1 the tool function is called directly instead of running Gemini.
"""

import asyncio
import os

from _schema import TEXTS, Ticket
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

import decisionsmith as ds
from decisionsmith.integrations.google_adk import tool

h = ds.harness(Ticket, teacher="gemini-2.5-flash", student="laya")
route = tool(h, name="route_ticket", description="Which team should handle this support ticket?")
agent = LlmAgent(
    name="support", model="gemini-2.5-flash", tools=[route], instruction="Route each ticket with route_ticket."
)


async def main():
    if os.environ.get("DS_OFFLINE"):
        for text in TEXTS:
            print(await route.func(text=text), "<-", text)
        return
    runner = InMemoryRunner(agent=agent, app_name="support")
    session = await runner.session_service.create_session(app_name="support", user_id="me")
    for text in TEXTS:
        message = types.Content(role="user", parts=[types.Part(text=text)])
        async for event in runner.run_async(user_id="me", session_id=session.id, new_message=message):
            if event.is_final_response() and event.content:
                print(event.content.parts[0].text)


asyncio.run(main())
