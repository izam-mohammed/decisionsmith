# LlamaIndex

Any LlamaIndex LLM as the teacher (OpenAI, Claude, Gemini, Bedrock, Azure, Mistral, Groq, Ollama, Hugging Face), a relevance filter for RAG, a router selector and a FunctionTool.

```python
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
```

| file | what it shows |
|---|---|
| [`in_framework_relevance_filter.py`](in_framework_relevance_filter.py) | A RAG relevance filter: retrieved nodes the model marks off-topic never reach the LLM. |
| [`in_framework_selector.py`](in_framework_selector.py) | A router selector: the harness picks which query engine answers, instead of an LLM call per query. |
| [`in_framework_tool.py`](in_framework_tool.py) | A `FunctionTool` for LlamaIndex agents: the agent calls the harness to route a ticket. |
| [`teacher_anthropic.py`](teacher_anthropic.py) | Claude through LlamaIndex (ANTHROPIC_API_KEY; pip install llama-index-llms-anthropic). |
| [`teacher_azure_openai.py`](teacher_azure_openai.py) | Azure OpenAI through LlamaIndex (AZURE_OPENAI_API_KEY; pip install llama-index-llms-azure-openai). |
| [`teacher_bedrock.py`](teacher_bedrock.py) | Amazon Nova on Bedrock through LlamaIndex (AWS credentials; pip install llama-index-llms-bedrock-converse). |
| [`teacher_gemini.py`](teacher_gemini.py) | Gemini through LlamaIndex's Google GenAI LLM (GOOGLE_API_KEY; pip install llama-index-llms-google-genai). |
| [`teacher_groq.py`](teacher_groq.py) | Groq through LlamaIndex (GROQ_API_KEY; pip install llama-index-llms-groq). |
| [`teacher_huggingface.py`](teacher_huggingface.py) | Hugging Face Inference Providers through LlamaIndex (HF_TOKEN; pip install llama-index-llms-huggingface-api). |
| [`teacher_mistral.py`](teacher_mistral.py) | Mistral through LlamaIndex (MISTRAL_API_KEY; pip install llama-index-llms-mistralai). |
| [`teacher_ollama.py`](teacher_ollama.py) | A local Ollama model through LlamaIndex (ollama pull qwen3; pip install llama-index-llms-ollama). |
| [`teacher_openai.py`](teacher_openai.py) | OpenAI through LlamaIndex (OPENAI_API_KEY). |

## Run

```bash
pip install "decisionsmith[llamaindex,laya]"
pip install llama-index-llms-openai llama-index-llms-anthropic llama-index-llms-google-genai llama-index-llms-bedrock-converse llama-index-llms-azure-openai llama-index-llms-mistralai llama-index-llms-groq llama-index-llms-ollama llama-index-llms-huggingface-api
export AZURE_OPENAI_API_KEY=...
export HF_TOKEN=...
python examples/04-integrations/llamaindex/in_framework_relevance_filter.py
python examples/04-integrations/llamaindex/in_framework_selector.py
python examples/04-integrations/llamaindex/in_framework_tool.py
python examples/04-integrations/llamaindex/teacher_anthropic.py
python examples/04-integrations/llamaindex/teacher_azure_openai.py
python examples/04-integrations/llamaindex/teacher_bedrock.py
python examples/04-integrations/llamaindex/teacher_gemini.py
python examples/04-integrations/llamaindex/teacher_groq.py
python examples/04-integrations/llamaindex/teacher_huggingface.py
python examples/04-integrations/llamaindex/teacher_mistral.py
python examples/04-integrations/llamaindex/teacher_ollama.py
python examples/04-integrations/llamaindex/teacher_openai.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
