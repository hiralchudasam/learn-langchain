# 18 — LangSmith

## What is LangSmith?

LangSmith is Anthropic's observability platform for LLM applications. It gives you:
- **Tracing** — see every LLM call, tool invocation, and chain step
- **Debugging** — inspect inputs/outputs at every step
- **Evaluation** — test your chain on datasets
- **Prompt Hub** — version and share prompts
- **Monitoring** — production dashboards

## Setup

```bash
pip install langsmith
```

```bash
# .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=my-project-name  # creates automatically
```

That's it — every LangChain call is now traced automatically.

## Viewing Traces

Go to [smith.langchain.com](https://smith.langchain.com) → your project → see all runs with full input/output trees.

## Creating Datasets & Running Evals

```python
from langsmith import Client

client = Client()

# Create a dataset
dataset = client.create_dataset("QA Test Set")
client.create_examples(
    inputs=[{"question": "What is RAG?"}],
    outputs=[{"answer": "Retrieval Augmented Generation"}],
    dataset_id=dataset.id,
)

# Run evaluation
from langchain.smith import run_on_dataset
results = run_on_dataset(
    client=client,
    dataset_name="QA Test Set",
    llm_or_chain_factory=lambda: my_chain,
    evaluation={"evaluators": ["qa"]},
)
```

## @traceable Decorator

Trace any Python function, not just LangChain components:

```python
from langsmith import traceable

@traceable(name="my_custom_step")
def preprocess_query(query: str) -> str:
    return query.strip().lower()
```

## Next Topic
→ [19 — LangServe](../19_langserve/README.md)
