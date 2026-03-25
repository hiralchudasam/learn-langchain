# 11 — Vector Stores

## What is a Vector Store?

A vector store (or vector database) stores embeddings alongside their source documents and lets you query them using **semantic similarity search** — finding documents whose meaning is closest to your query.

```
Store:  embed(doc) → store (vector, metadata, text)
Search: embed(query) → find closest vectors → return matching docs
```

---

## Common Vector Stores

| Name | Type | Best For |
|------|------|---------|
| **Chroma** | Local / persistent | Development, learning |
| **FAISS** | In-memory / file | Large-scale local search |
| **Pinecone** | Cloud, managed | Production, scale |
| **Qdrant** | Self-hosted / cloud | Production, filtering |
| **Weaviate** | Self-hosted / cloud | Hybrid search |
| **pgvector** | PostgreSQL extension | Already using Postgres |
| **Milvus** | Self-hosted | Billion-scale |

---

## Chroma (Local — Great for Learning)

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Create from documents
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db",   # saves to disk
    collection_name="my_docs",
)

# Load existing
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(),
)
```

---

## Core Operations

```python
# Add documents
vectorstore.add_documents(new_docs)

# Similarity search (returns Documents)
docs = vectorstore.similarity_search("What is RAG?", k=4)

# With scores (returns list of (Document, float) tuples)
docs_scores = vectorstore.similarity_search_with_score("What is RAG?", k=4)

# MMR — Maximal Marginal Relevance (reduces redundancy)
docs = vectorstore.max_marginal_relevance_search("What is RAG?", k=4, fetch_k=20)

# As a retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",        # or "mmr", "similarity_score_threshold"
    search_kwargs={"k": 4},
)
```

---

## Filtering with Metadata

```python
# Add docs with metadata
from langchain_core.documents import Document

docs = [
    Document(page_content="RAG overview", metadata={"source": "doc1", "year": 2024}),
    Document(page_content="Agent overview", metadata={"source": "doc2", "year": 2023}),
]

# Filter by metadata during search
results = vectorstore.similarity_search(
    "RAG",
    k=2,
    filter={"year": 2024},           # Only return docs from 2024
)
```

---

## FAISS (Fast In-Memory)

```python
from langchain_community.vectorstores import FAISS

# Create
db = FAISS.from_documents(docs, embeddings)

# Save / Load
db.save_local("faiss_index")
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Merge two FAISS indexes
db1.merge_from(db2)
```

---

## Next Topic

→ [12 — Retrievers](../12_retrievers/README.md)
