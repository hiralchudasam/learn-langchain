"""
09 + 10 — Text Splitters & Embeddings
Example 01: Split text, generate embeddings, compute similarity
"""

from dotenv import load_dotenv
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import numpy as np

load_dotenv()

SAMPLE_TEXT = """
LangChain is an open-source framework for developing applications powered by language models.
It provides a standard interface for chains, agents, and retrieval strategies.
LangChain was founded by Harrison Chase in 2022 and quickly became one of the most popular AI frameworks.

LCEL (LangChain Expression Language) is the modern way to compose chains using the pipe operator.
It supports streaming, async, batching, and tracing out of the box.
LCEL replaced the older imperative chain construction style.

Agents in LangChain use LLMs to decide which tools to call.
The ReAct pattern (Reason + Act) is the most common agent strategy.
LangGraph extends this with stateful graph-based orchestration.

RAG (Retrieval Augmented Generation) combines vector search with LLM generation.
It allows LLMs to answer questions about private, real-time, or specialized data.
Common vector stores include Chroma, FAISS, Pinecone, and Weaviate.
"""

# ─── 1. RecursiveCharacterTextSplitter ────────────────────────────────────────

print("=" * 55)
print("1. RecursiveCharacterTextSplitter")
print("=" * 55)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = splitter.split_text(SAMPLE_TEXT)
print(f"Input length: {len(SAMPLE_TEXT)} chars")
print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1} ({len(chunk)} chars):\n  {chunk[:100]}...")

# ─── 2. Split Documents (with metadata) ───────────────────────────────────────

print("\n" + "=" * 55)
print("2. Splitting Documents (preserves metadata)")
print("=" * 55)

docs = [
    Document(page_content=SAMPLE_TEXT, metadata={"source": "langchain_guide.txt", "chapter": 1}),
]

doc_chunks = splitter.split_documents(docs)
print(f"Produced {len(doc_chunks)} document chunks")
for chunk in doc_chunks[:2]:
    print(f"  Metadata: {chunk.metadata}")
    print(f"  Content: {chunk.page_content[:80]}...")

# ─── 3. MarkdownHeaderTextSplitter ────────────────────────────────────────────

print("\n" + "=" * 55)
print("3. MarkdownHeaderTextSplitter")
print("=" * 55)

MARKDOWN = """
# Introduction to LangChain

LangChain is a framework for building LLM applications.

## Core Concepts

### Chains
A chain is a sequence of calls to models, prompts, and parsers.

### Agents
Agents use LLMs to decide which tools to call dynamically.

## Advanced Topics

### RAG
RAG combines vector search with LLM generation for grounded answers.
"""

headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
md_chunks = md_splitter.split_text(MARKDOWN)

for chunk in md_chunks:
    print(f"  Metadata: {chunk.metadata}")
    print(f"  Content: {chunk.page_content[:60]}...\n")

# ─── 4. Embeddings ────────────────────────────────────────────────────────────

print("=" * 55)
print("4. Generating Embeddings")
print("=" * 55)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

texts = [
    "LangChain is a framework for LLM applications",
    "Python is a general-purpose programming language",
    "Machine learning is a subset of artificial intelligence",
    "LangChain provides tools for building AI agents",
]

vectors = embeddings.embed_documents(texts)
print(f"Embedded {len(vectors)} texts")
print(f"Each vector has {len(vectors[0])} dimensions")

# ─── 5. Semantic Similarity ───────────────────────────────────────────────────

print("\n" + "=" * 55)
print("5. Semantic Similarity")
print("=" * 55)

def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

query = "AI development tools"
query_vector = embeddings.embed_query(query)

print(f"Query: '{query}'")
print("\nSimilarity to each text:")
for text, vec in zip(texts, vectors):
    score = cosine_similarity(query_vector, vec)
    bar = "█" * int(score * 30)
    print(f"  {score:.3f} {bar} | {text[:50]}")

# Most similar
scores = [(cosine_similarity(query_vector, v), t) for v, t in zip(vectors, texts)]
best_score, best_text = max(scores)
print(f"\nMost similar: '{best_text}' (score: {best_score:.3f})")
