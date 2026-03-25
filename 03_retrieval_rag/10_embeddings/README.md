# 10 — Embeddings

## What are Embeddings?

An **embedding** is a list of floating-point numbers that represents the *meaning* of a piece of text. Texts with similar meanings have vectors that are numerically close to each other.

```
"I love cats"  → [0.021, -0.034, 0.198, ...]  ← 1536 numbers
"I adore cats" → [0.019, -0.031, 0.201, ...]  ← close! (similar meaning)
"Quantum physics" → [-0.412, 0.887, -0.023, ...]  ← far away (different meaning)
```

---

## Two Types of Embedding Calls

```python
# embed_query — for search queries (single string)
query_vector = embeddings.embed_query("What is RAG?")

# embed_documents — for documents to store (list of strings)
doc_vectors = embeddings.embed_documents(["Doc 1...", "Doc 2..."])
```

---

## Supported Embedding Models

### OpenAI Embeddings (most popular)
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # 1536 dims, cheap
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # 3072 dims, better
```

| Model | Dimensions | Cost |
|-------|-----------|------|
| text-embedding-3-small | 1536 | $0.02 / 1M tokens |
| text-embedding-3-large | 3072 | $0.13 / 1M tokens |
| text-embedding-ada-002 | 1536 | $0.10 / 1M tokens (legacy) |

### HuggingFace Embeddings (free, local)
```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# Runs locally, no API key needed
```

### Cohere Embeddings
```python
from langchain_cohere import CohereEmbeddings
embeddings = CohereEmbeddings(model="embed-english-v3.0")
```

### Ollama (fully local)
```python
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

---

## Caching Embeddings

Embedding every document on every run wastes time and money. Cache them:

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

store = LocalFileStore("./embedding_cache")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=OpenAIEmbeddings(),
    document_embedding_cache=store,
)
# First call: hits OpenAI API
# Second call: reads from disk cache instantly
```

---

## Similarity Search

Cosine similarity is used to compare vectors:

```python
import numpy as np

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
# Returns -1 to 1. Closer to 1 = more similar.
```

---

## Choosing an Embedding Model

| Need | Recommendation |
|------|----------------|
| Best quality | `text-embedding-3-large` |
| Cost-effective | `text-embedding-3-small` |
| Free / offline | `all-MiniLM-L6-v2` (HuggingFace) |
| Multilingual | `multilingual-e5-large` |

---

## Next Topic

→ [11 — Vector Stores](../11_vector_stores/README.md)
