"""A router selector: the harness picks which query engine answers, instead of an LLM call per query."""

from _schema import TEXTS, Ticket
from llama_index.core.query_engine import CustomQueryEngine, RouterQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

import decisionsmith as ds
from decisionsmith.integrations.llamaindex import selector


class TeamDocs(CustomQueryEngine):
    """Stands in for each team's own index (index.as_query_engine())."""

    team: str

    def custom_query(self, query_str: str) -> str:
        return "[%s docs] %s" % (self.team, query_str)


tools = [
    QueryEngineTool(query_engine=TeamDocs(team=t), metadata=ToolMetadata(name=t, description="%s questions" % t))
    for t in ["billing", "technical", "sales"]
]
h = ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya")
router = RouterQueryEngine(selector=selector(h, field="team"), query_engine_tools=tools)

for text in TEXTS[:3]:
    print(router.query(text))
