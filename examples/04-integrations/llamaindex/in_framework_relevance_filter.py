"""A RAG relevance filter: retrieved nodes the model marks off-topic never reach the LLM.

In a query engine: index.as_query_engine(node_postprocessors=[keep]). Train the model on your own
query/passage pairs first; base Laya is only a starting point.
"""

from llama_index.core.schema import NodeWithScore, TextNode

import decisionsmith as ds
from decisionsmith.integrations.llamaindex import relevance_filter

judge = ds.model(["relevant", "off-topic"], question="Does the passage help answer the query?")
keep = relevance_filter(judge, keep=["relevant"])

retrieved = [
    NodeWithScore(node=TextNode(text=t), score=0.8)
    for t in [
        "Refunds are paid back to the original card within 5 business days.",
        "Our office dog is called Biscuit.",
        "To request a refund, reply to your receipt email.",
    ]
]
for n in keep.postprocess_nodes(retrieved, query_str="How do I get a refund?"):
    print("kept:", n.node.get_content())
