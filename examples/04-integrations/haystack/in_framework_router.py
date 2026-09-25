"""A pipeline router: each ticket goes to its team's branch; the harness decides, no LLM call per ticket once the
student is sure. `router(h, "team")` has one output per team: route.billing, route.technical, route.sales.
"""

from _schema import TEXTS, Ticket
from haystack import Pipeline, component

import decisionsmith as ds
from decisionsmith.integrations.haystack import router


@component
class TeamQueue:
    """Stands in for each team's own branch (a prompt builder + generator, a ticketing API, ...)."""

    def __init__(self, team: str) -> None:
        self.team = team

    @component.output_types(queued=str)
    def run(self, text: str) -> dict:
        return {"queued": "[%s] %s" % (self.team, text)}


h = ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya")
pipe = Pipeline()
pipe.add_component("route", router(h, field="team"))
for team in ["billing", "technical", "sales"]:
    pipe.add_component(team, TeamQueue(team))
    pipe.connect("route.%s" % team, "%s.text" % team)

for text in TEXTS[:3]:
    for result in pipe.run({"route": {"text": text}}).values():
        print(result["queued"])
