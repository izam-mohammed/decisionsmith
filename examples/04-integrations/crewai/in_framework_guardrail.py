"""A task guardrail: your model reads the agent's answer, and CrewAI asks again when it promises a refund
(OPENAI_API_KEY).

With DS_OFFLINE=1 the guardrail is called on two task outputs instead of kicking off the crew.
"""

import os

from _schema import Reply
from crewai import Agent, Crew, Task
from crewai.tasks.task_output import TaskOutput

import decisionsmith as ds
from decisionsmith.integrations.crewai import guardrail

h = ds.harness(Reply, teacher="gpt-5-mini", student="laya")
check = guardrail(h, "promises_refund", block=[True], message="Do not promise a refund; answer again.")

if os.environ.get("DS_OFFLINE"):
    for raw in ["Thanks, our billing team will look into the double charge.", "Yes, we will refund you today."]:
        ok, _ = check(TaskOutput(description="reply", raw=raw, agent="support"))
        print("pass:" if ok else "retry:", raw)
else:
    agent = Agent(
        role="Support agent", goal="Answer customers politely", backstory="Never promise refunds.", llm="gpt-5-mini"
    )
    task = Task(
        description="Reply to: {message}",
        expected_output="A short reply",
        agent=agent,
        guardrail=check,
        guardrail_max_retries=2,
    )
    print(Crew(agents=[agent], tasks=[task]).kickoff(inputs={"message": "I was charged twice, refund me"}).raw)
