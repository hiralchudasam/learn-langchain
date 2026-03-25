"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 10 — Embeddings                                       ║
║  File:   01_embeddings.py                                    ║
║  Level:  Basic → Advanced                                    ║
║  Goal:   Generate embeddings, compute similarity, compare    ║
║          providers, and cache for cost savings               ║
╚══════════════════════════════════════════════════════════════╝
"""

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from langchain_openai import OpenAIEmbeddings

print("=" * 60)
print("10 — Embeddings")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# SECTION 1: What are embeddings?
# An embedding is a list of floats that encodes the MEANING of text.
# Texts with similar meanings → similar vectors (close in space).
# This enables semantic search — find "cat" when user types "feline".
# ─────────────────────────────────────────────────────────────
print("\n📌 1. Generating Embeddings")
print("─" * 40)

embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# embed_query → for search queries (single string)
query_vec = embedder.embed_query("What is LangChain?")
print(f"  Input       : 'What is LangChain?'")
print(f"  Vector size : {len(query_vec)} dimensions")
print(f"  First 5 vals: {[round(v, 4) for v in query_vec[:5]]}")
print(f"  Type        : {type(query_vec[0]).__name__} (each value is a float)")

# embed_documents → for a batch of documents
docs = [
    "LangChain is a framework for building LLM applications.",
    "The Eiffel Tower is located in Paris, France.",
    "Python is a general-purpose programming language.",
    "LangGraph extends LangChain with stateful agent graphs.",
    "Football is the most popular sport in the world.",
]

doc_vecs = embedder.embed_documents(docs)
print(f"\n  Embedded {len(doc_vecs)} documents")
print(f"  Each vector: {len(doc_vecs[0])} dimensions")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Cosine Similarity — the core search mechanism
# Higher score (closer to 1.0) = more similar meaning.
# This is how vector stores find relevant documents.
# ─────────────────────────────────────────────────────────────
print("\n📌 2. Semantic Similarity")
print("─" * 40)

def cosine_similarity(v1: list, v2: list) -> float:
    """Compute cosine similarity between two vectors. Range: -1 to 1."""
    a, b = np.array(v1), np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def semantic_search(query: str, corpus: list[str], top_k: int = 3) -> list:
    """Find the most semantically similar texts to a query."""
    q_vec     = embedder.embed_query(query)
    doc_vecs  = embedder.embed_documents(corpus)
    scores    = [(cosine_similarity(q_vec, dv), doc) for dv, doc in zip(doc_vecs, corpus)]
    return sorted(scores, reverse=True)[:top_k]

# Test semantic search
queries = [
    "Tell me about AI frameworks",
    "European landmarks",
    "programming languages",
]

for query in queries:
    results = semantic_search(query, docs)
    print(f"\n  Query: '{query}'")
    for score, doc in results:
        bar = "█" * int(score * 25)
        print(f"    {score:.3f} {bar:<25} {doc[:55]}")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Similarity vs dissimilarity
# See how similar/different pairs score vs each other.
# ─────────────────────────────────────────────────────────────
print("\n📌 3. Similarity vs Dissimilarity Examples")
print("─" * 40)

pairs = [
    ("I love programming",       "I enjoy coding",            "Very similar (synonyms)"),
    ("Python is great",          "Python snake is dangerous",  "Ambiguous (same word, different meaning)"),
    ("LangChain builds LLM apps","The sky is blue",            "Completely unrelated"),
    ("cat",                      "kitten",                     "Related (same animal)"),
    ("cat",                      "dog",                        "Related (both animals)"),
    ("cat",                      "quantum physics",            "Unrelated"),
]

for text1, text2, description in pairs:
    v1    = embedder.embed_query(text1)
    v2    = embedder.embed_query(text2)
    score = cosine_similarity(v1, v2)
    print(f"  {score:.3f}  '{text1[:25]:<25}' vs '{text2[:25]:<25}'  ({description})")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Embedding Caching — save money + time
# Caching prevents re-embedding the same text twice.
# In production this can save 90%+ of embedding API costs.
# ─────────────────────────────────────────────────────────────
print("\n📌 4. Embedding Cache")
print("─" * 40)

import time
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

store           = LocalFileStore("./embedding_cache_demo")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embedder,
    document_embedding_cache=store,
    namespace=embedder.model,
)

texts_to_embed = ["What is RAG?", "Explain LangGraph.", "What are agents?"]

# First run — hits the API
start = time.time()
vecs1 = cached_embedder.embed_documents(texts_to_embed)
t1    = time.time() - start
print(f"  First run  (API call)  : {t1:.3f}s")

# Second run — reads from cache
start = time.time()
vecs2 = cached_embedder.embed_documents(texts_to_embed)
t2    = time.time() - start
print(f"  Second run (cache hit) : {t2:.3f}s")
print(f"  Speedup                : {t1/max(t2, 0.0001):.0f}x faster!")
print(f"  Vectors identical      : {vecs1[0][:3] == vecs2[0][:3]}")

# Cleanup cache
import shutil
shutil.rmtree("./embedding_cache_demo", ignore_errors=True)

# ─────────────────────────────────────────────────────────────
# SECTION 5: Embedding model comparison
# Different models trade off cost, quality, and size.
# ─────────────────────────────────────────────────────────────
print("\n📌 5. Embedding Model Comparison")
print("─" * 40)

models = {
    "text-embedding-3-small": OpenAIEmbeddings(model="text-embedding-3-small"),
    "text-embedding-3-large": OpenAIEmbeddings(model="text-embedding-3-large"),
}

query = "LangChain RAG pipeline"
test_docs_similar    = ["LangChain retrieval-augmented generation", "Building RAG with LangChain"]
test_docs_dissimilar = ["The weather is sunny today",               "I like playing football"]

print(f"  Query: '{query}'")
for model_name, emb in models.items():
    q_vec = emb.embed_query(query)
    dim   = len(q_vec)

    similar_vecs    = emb.embed_documents(test_docs_similar)
    dissimilar_vecs = emb.embed_documents(test_docs_dissimilar)

    sim1 = cosine_similarity(q_vec, similar_vecs[0])
    sim2 = cosine_similarity(q_vec, dissimilar_vecs[0])
    gap  = sim1 - sim2  # bigger gap = better discrimination

    print(f"\n  [{model_name}]")
    print(f"    Dimensions    : {dim}")
    print(f"    Similar score : {sim1:.3f}")
    print(f"    Unrelated score: {sim2:.3f}")
    print(f"    Gap           : {gap:.3f}  (higher = better discrimination)")

# ─────────────────────────────────────────────────────────────
# ⚠️  COMMON MISTAKE 1: Using embed_query() for documents
# embed_query()     → optimized for search queries (single text)
# embed_documents() → optimized for corpus documents (batch)
# Some models use different encodings internally — use the right one.

# ⚠️  COMMON MISTAKE 2: Not caching embeddings in production
# Embedding the same docs on every restart is wasteful.
# Always use CacheBackedEmbeddings or persist your vector store.

# ⚠️  COMMON MISTAKE 3: Assuming high similarity = correct answer
# Cosine similarity measures meaning-closeness, not factual accuracy.
# A retrieved doc may be semantically close but contain wrong info.
# ─────────────────────────────────────────────────────────────
print("\n⚠️  Embeddings tips:")
print("   • Use embed_query() for queries, embed_documents() for corpus")
print("   • Always cache embeddings — saves 90%+ cost in production")
print("   • text-embedding-3-small is best for cost/quality balance")
print("   • High similarity ≠ factual accuracy — always verify answers")
