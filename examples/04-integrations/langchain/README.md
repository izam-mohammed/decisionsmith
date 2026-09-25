# LangChain

Any LangChain chat model as the teacher (OpenAI, Claude, Gemini, Vertex, Bedrock, Azure, Mistral, Groq, Ollama, Hugging Face, Fireworks, Together), and a DecisionRunnable, tool, output parser and document compressor.

```python
"""RAG relevance filtering: a document compressor keeps the retrieved documents the model finds relevant.

Use it as `ContextualCompressionRetriever(base_compressor=keep, base_retriever=retriever)` (langchain-classic),
or call it on any list of documents as below (OPENAI_API_KEY for the teacher).
"""

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import decisionsmith as ds
from decisionsmith.integrations.langchain import compressor


class Relevance(BaseModel):
    relevant: bool = Field(description="Does the document help answer the query?")


h = ds.harness(Relevance, teacher=ChatOpenAI(model="gpt-5-mini"), student="laya")
keep = compressor(h, "relevant", keep=[True])

retrieved = [
    Document("Refunds are paid back to the original card within 5 business days."),
    Document("Our office is closed on public holidays."),
    Document("Duplicate charges are reversed automatically after review."),
]
for doc in keep.compress_documents(retrieved, "I was charged twice, when do I get my money back?"):
    print("keep:", doc.page_content)
```

| file | what it shows |
|---|---|
| [`in_framework_compressor.py`](in_framework_compressor.py) | RAG relevance filtering: a document compressor keeps the retrieved documents the model finds relevant. |
| [`in_framework_output_parser.py`](in_framework_output_parser.py) | An output parser: an LLM writes free text, the harness turns it into a decision (`prompt | llm | parser`). |
| [`in_framework_runnable.py`](in_framework_runnable.py) | A DecisionRunnable in a chain: invoke, batch and `|` like any LangChain Runnable (OPENAI_API_KEY for the teacher). |
| [`in_framework_tool.py`](in_framework_tool.py) | A decision as a LangChain tool: bind it to any chat model that calls tools (OPENAI_API_KEY for the teacher). |
| [`teacher_anthropic.py`](teacher_anthropic.py) | Claude through ChatAnthropic (ANTHROPIC_API_KEY). |
| [`teacher_azure_openai.py`](teacher_azure_openai.py) | Azure OpenAI through AzureChatOpenAI (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT). |
| [`teacher_bedrock.py`](teacher_bedrock.py) | Amazon Nova on Bedrock through ChatBedrockConverse (AWS credentials; pick your region). |
| [`teacher_fireworks.py`](teacher_fireworks.py) | Fireworks through ChatFireworks (FIREWORKS_API_KEY). |
| [`teacher_google_genai.py`](teacher_google_genai.py) | Gemini (Google AI Studio) through ChatGoogleGenerativeAI (GOOGLE_API_KEY). |
| [`teacher_groq.py`](teacher_groq.py) | Groq through ChatGroq (GROQ_API_KEY). |
| [`teacher_huggingface.py`](teacher_huggingface.py) | Hugging Face Inference Providers through ChatHuggingFace (HF_TOKEN). |
| [`teacher_mistral.py`](teacher_mistral.py) | Mistral through ChatMistralAI (MISTRAL_API_KEY). |
| [`teacher_ollama.py`](teacher_ollama.py) | A local Ollama model through ChatOllama (ollama pull qwen3); no key, nothing leaves the machine. |
| [`teacher_openai.py`](teacher_openai.py) | OpenAI through ChatOpenAI (OPENAI_API_KEY). |
| [`teacher_together.py`](teacher_together.py) | Together AI through ChatTogether (TOGETHER_API_KEY). |
| [`teacher_vertex.py`](teacher_vertex.py) | Gemini on Vertex AI through ChatGoogleGenerativeAI (gcloud auth application-default login; GOOGLE_CLOUD_PROJECT). |

## Run

```bash
uv add "decisionsmith[langchain,laya]"
uv add langchain-openai langchain-anthropic langchain-google-genai langchain-aws langchain-mistralai langchain-groq langchain-ollama langchain-huggingface langchain-fireworks langchain-together
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=...
export GOOGLE_CLOUD_PROJECT=...
export HF_TOKEN=...
uv run python examples/04-integrations/langchain/in_framework_compressor.py
uv run python examples/04-integrations/langchain/in_framework_output_parser.py
uv run python examples/04-integrations/langchain/in_framework_runnable.py
uv run python examples/04-integrations/langchain/in_framework_tool.py
uv run python examples/04-integrations/langchain/teacher_anthropic.py
uv run python examples/04-integrations/langchain/teacher_azure_openai.py
uv run python examples/04-integrations/langchain/teacher_bedrock.py
uv run python examples/04-integrations/langchain/teacher_fireworks.py
uv run python examples/04-integrations/langchain/teacher_google_genai.py
uv run python examples/04-integrations/langchain/teacher_groq.py
uv run python examples/04-integrations/langchain/teacher_huggingface.py
uv run python examples/04-integrations/langchain/teacher_mistral.py
uv run python examples/04-integrations/langchain/teacher_ollama.py
uv run python examples/04-integrations/langchain/teacher_openai.py
uv run python examples/04-integrations/langchain/teacher_together.py
uv run python examples/04-integrations/langchain/teacher_vertex.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
