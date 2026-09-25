"""An output parser: an LLM writes free text, the harness turns it into a decision (`prompt | llm | parser`).

With DS_OFFLINE=1 a canned chat model stands in for Claude so this runs without a key (ANTHROPIC_API_KEY).
"""

import os

from _schema import Ticket
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate

import decisionsmith as ds
from decisionsmith.integrations.langchain import output_parser

h = ds.harness(Ticket, teacher=ChatAnthropic(model="claude-haiku-4-5"), student="laya")
llm = ChatAnthropic(model="claude-haiku-4-5")
if os.environ.get("DS_OFFLINE"):
    llm = FakeListChatModel(responses=["The customer was billed twice and wants the extra charge refunded."])

prompt = ChatPromptTemplate.from_messages([("human", "Summarise this call transcript in one sentence:\n{call}")])
chain = prompt | llm | output_parser(h)
print(chain.invoke({"call": "Hi, yes, I looked at my card and you took the payment twice this month..."}))
