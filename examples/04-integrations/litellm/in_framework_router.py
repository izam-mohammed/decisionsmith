"""LiteLLM Router tiers: a decision picks the cheap or the strong model group for each request.

`mock_response` keeps this runnable without keys; drop it to call the real models.
"""

import litellm

import decisionsmith as ds
from decisionsmith.integrations.litellm import tier

router = litellm.Router(
    model_list=[
        {"model_name": "cheap", "litellm_params": {"model": "gpt-5-mini", "mock_response": "(cheap model reply)"}},
        {"model_name": "strong", "litellm_params": {"model": "gpt-5", "mock_response": "(strong model reply)"}},
    ]
)
hard = ds.model(["simple", "hard"], question="Does answering this need careful multi-step reasoning?")
pick = tier(hard, None, {"simple": "cheap", "hard": "strong"})

for text in ["What time do you open?", "Plan a migration of our billing data to a new schema without downtime"]:
    messages = [{"role": "user", "content": text}]
    group = pick(messages)
    reply = router.completion(model=group, messages=messages)
    print(group, "->", reply.choices[0].message.content)
