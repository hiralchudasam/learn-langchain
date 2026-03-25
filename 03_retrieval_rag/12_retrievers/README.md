# 12 — Retrievers

## What is a Retriever?

A retriever is any object that accepts a string query and returns a list of `Document` objects. It's the standard interface for fetching relevant context in a RAG pipeline.

```python
docs: list[Document] = retriever.invoke("What is LangChain?")
```

---

## Types of Retrievers

### VectorStoreRetriever
The most basic retriever — wraps a vector store.
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

### MultiQueryRetriever
Generates multiple query variants using an LLM, retrieves for each, and deduplicates. Helps when the user's query is ambiguous or short.
```python
from langchain.retrievers.multi_query import MultiQueryRetriever
retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
```

### ContextualCompressionRetriever
First retrieves, then compresses/filters the docs using an LLM to only keep the relevant parts.
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)
```

### ParentDocumentRetriever
Stores large parent chunks; embeds small child chunks. Retrieves small chunks but returns full parent. Best of both worlds: precise retrieval + full context.
```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
```

### SelfQueryRetriever
Lets the LLM generate both a query string AND metadata filters from a natural language question.
```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
# "Show me AI papers from 2024" → query="AI" + filter={"year": 2024}
```

### EnsembleRetriever (Hybrid Search)
Combines multiple retrievers using Reciprocal Rank Fusion.
```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(docs)       # keyword search
vector = vectorstore.as_retriever()              # semantic search

ensemble = EnsembleRetriever(
    retrievers=[bm25, vector],
    weights=[0.5, 0.5],
)
# Best of both: finds exact keywords AND semantic matches
```

---

## Retriever as a Runnable

All retrievers implement the Runnable interface:
```python
retriever.invoke("query")           # returns list[Document]
retriever.batch(["q1", "q2"])       # parallel
retriever.stream("query")           # stream results
```

---

## Choosing a Retriever

| Situation | Best Retriever |
|-----------|---------------|
| Simple RAG | VectorStoreRetriever |
| Vague/short queries | MultiQueryRetriever |
| Reduce noise in docs | ContextualCompressionRetriever |
| Need full context | ParentDocumentRetriever |
| Metadata filtering from NL | SelfQueryRetriever |
| Keyword + semantic mix | EnsembleRetriever |

---

## Next Topic

→ [13 — RAG Patterns](../13_rag_patterns/README.md)
