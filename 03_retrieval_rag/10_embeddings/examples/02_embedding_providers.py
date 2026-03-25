"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 10 — Embeddings                                       ║
║  File:   02_embedding_providers.py                           ║
║  Level:  Intermediate                                        ║
║  Goal:   HuggingFace local embeddings, caching strategies,  ║
║          and building a mini semantic search engine          ║
╚══════════════════════════════════════════════════════════════╝
"""

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

print("=" * 60)
print("10 — Embedding Providers & Search")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# Provider comparison: OpenAI vs HuggingFace (local)
# OpenAI: paid API, very high quality, 1536 dims
# HuggingFace: free, local, ~384 dims, slightly lower quality
# ─────────────────────────────────────────────────────────────
print("\n📌 1. OpenAI vs Local HuggingFace Embeddings")
print("─" * 40)

openai_embedder = OpenAIEmbeddings(model="text-embedding-3-small")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    hf_embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    hf_available = True
except ImportError:
    hf_available = False
    print("  HuggingFace not installed: pip install langchain-huggingface sentence-transformers")

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

pairs = [
    ("LangChain is an LLM framework", "Framework for building LLM apps"),
    ("LangChain is an LLM framework", "The sun is a star"),
]

for text1, text2 in pairs:
    openai_sim = cosine_sim(
        openai_embedder.embed_query(text1),
        openai_embedder.embed_query(text2),
    )
    row = f"  OpenAI: {openai_sim:.3f}"

    if hf_available:
        hf_sim = cosine_sim(
            hf_embedder.embed_query(text1),
            hf_embedder.embed_query(text2),
        )
        row += f"  | HF: {hf_sim:.3f}"

    print(f"  '{text1[:35]}'")
    print(f"  '{text2[:35]}'")
    print(f"  {row}\n")

# ─────────────────────────────────────────────────────────────
# Mini semantic search engine
# ─────────────────────────────────────────────────────────────
print("\n📌 2. Mini Semantic Search Engine")
print("─" * 40)

KNOWLEDGE_BASE = [
    "Python is a high-level programming language known for its readability.",
    "LangChain provides building blocks for LLM-powered applications.",
    "Vector databases store embeddings and support semantic search.",
    "RAG combines information retrieval with language model generation.",
    "Neural networks are computing systems inspired by biological brains.",
    "LangGraph extends LangChain with stateful graph orchestration.",
    "FastAPI is a modern Python web framework for building APIs.",
    "Transformers use attention mechanisms to process sequential data.",
]

# Pre-embed the knowledge base (do this once, cache the result)
print("  Embedding knowledge base...")
kb_vectors = openai_embedder.embed_documents(KNOWLEDGE_BASE)

def search(query: str, top_k: int = 3) -> list[tuple[float, str]]:
    q_vec  = openai_embedder.embed_query(query)
    scores = [(cosine_sim(q_vec, kv), doc) for kv, doc in zip(kb_vectors, KNOWLEDGE_BASE)]
    return sorted(scores, reverse=True)[:top_k]

queries = [
    "How do I build apps with AI?",
    "What is semantic search?",
    "Tell me about Python web development",
]

for q in queries:
    results = search(q, top_k=2)
    print(f"\n  Query: '{q}'")
    for score, doc in results:
        bar = "█" * int(score * 20)
        print(f"    {score:.3f} {bar:<20} {doc[:60]}")

# ─────────────────────────────────────────────────────────────
# Choosing the right model
# ─────────────────────────────────────────────────────────────
print("\n📌 3. Embedding Model Decision Guide")
print("─" * 40)

models = [
    ("text-embedding-3-small", "OpenAI", "1536d", "$0.02/1M",  "Best cost/quality balance"),
    ("text-embedding-3-large", "OpenAI", "3072d", "$0.13/1M",  "Highest quality"),
    ("all-MiniLM-L6-v2",       "HF",     "384d",  "Free/local","Fast, good for dev/testing"),
    ("multilingual-e5-large",  "HF",     "1024d", "Free/local","Best for non-English text"),
    ("nomic-embed-text",       "Ollama", "768d",  "Free/local","Good quality, fully offline"),
]

print(f"  {'Model':<30} {'Provider':<8} {'Dims':<6} {'Cost':<12} Use case")
print(f"  {'─'*30} {'─'*8} {'─'*6} {'─'*12} {'─'*30}")
for model, provider, dims, cost, use in models:
    print(f"  {model:<30} {provider:<8} {dims:<6} {cost:<12} {use}")

print("\n⚠️  Embedding tips:")
print("   • text-embedding-3-small is the best default choice")
print("   • Use HuggingFace models for free local development")
print("   • Multilingual models needed for non-English corpora")
print("   • Always pre-embed your corpus — don't embed on every query")
