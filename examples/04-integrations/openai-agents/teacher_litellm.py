"""Claude through the Agents SDK's LiteLLM model as the teacher (ANTHROPIC_API_KEY; openai-agents[litellm])."""

from _schema import TEXTS, Ticket
from agents.extensions.models.litellm_model import LitellmModel

import decisionsmith as ds
from decisionsmith.integrations.openai_agents import teacher

h = ds.harness(Ticket, teacher=teacher(LitellmModel("anthropic/claude-haiku-4-5")), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
