# Haystack

Any Haystack chat generator as the teacher (OpenAI, Claude, Gemini, Ollama, Hugging Face), and pipeline components backed by your model, a router with one output per value and a document filter for RAG.

```python
"""A RAG relevance filter: retrieved documents the model marks off-topic never reach the generator.

Put it between the retriever and the prompt builder. Train the model on your own query/document pairs first;
base Laya is only a starting point.
"""

from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

import decisionsmith as ds
from decisionsmith.integrations.haystack import document_filter

store = InMemoryDocumentStore()
store.write_documents(
    [
        Document(content="Refunds are paid back to the original card within 5 business days."),
        Document(content="Our office dog is called Biscuit and refuses to give refunds."),
        Document(content="To request a refund, reply to your receipt email."),
    ]
)
judge = ds.model(["relevant", "off-topic"], question="Does the document help answer the query?")

pipe = Pipeline()
pipe.add_component("retrieve", InMemoryBM25Retriever(store, top_k=3))
pipe.add_component("keep", document_filter(judge, keep=["relevant"]))
pipe.connect("retrieve.documents", "keep.documents")

query = "How do I get a refund?"
kept = pipe.run({"retrieve": {"query": query}, "keep": {"query": query}})["keep"]["documents"]
print(len(kept), "of 3 documents kept")
for doc in kept:
    print("kept:", doc.content)
```

| file | what it shows |
|---|---|
| [`in_framework_document_filter.py`](in_framework_document_filter.py) | A RAG relevance filter: retrieved documents the model marks off-topic never reach the generator. |
| [`in_framework_router.py`](in_framework_router.py) | A pipeline router: each ticket goes to its team's branch; the harness decides, no LLM call per ticket once the |
| [`teacher_anthropic.py`](teacher_anthropic.py) | Claude through Haystack (ANTHROPIC_API_KEY; uv add anthropic-haystack). |
| [`teacher_google_ai.py`](teacher_google_ai.py) | Gemini through Haystack's Google GenAI generator (GEMINI_API_KEY or GOOGLE_API_KEY; uv add |
| [`teacher_huggingface.py`](teacher_huggingface.py) | Hugging Face Inference Providers through Haystack (HF_TOKEN; uv add huggingface-api-haystack). |
| [`teacher_ollama.py`](teacher_ollama.py) | A local Ollama model through Haystack (ollama pull qwen3; uv add ollama-haystack). |
| [`teacher_openai.py`](teacher_openai.py) | OpenAI through Haystack's OpenAIChatGenerator (OPENAI_API_KEY). |

## Run

```bash
uv add "decisionsmith[haystack,laya]"
uv add anthropic-haystack google-genai-haystack ollama-haystack huggingface-api-haystack
export HF_TOKEN=...
uv run python examples/04-integrations/haystack/in_framework_document_filter.py
uv run python examples/04-integrations/haystack/in_framework_router.py
uv run python examples/04-integrations/haystack/teacher_anthropic.py
uv run python examples/04-integrations/haystack/teacher_google_ai.py
uv run python examples/04-integrations/haystack/teacher_huggingface.py
uv run python examples/04-integrations/haystack/teacher_ollama.py
uv run python examples/04-integrations/haystack/teacher_openai.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
