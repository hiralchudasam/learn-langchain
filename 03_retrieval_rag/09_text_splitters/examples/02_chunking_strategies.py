"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 09 — Text Splitters                                   ║
║  File:   02_chunking_strategies.py                           ║
║  Level:  Advanced                                            ║
║  Goal:   Compare chunking strategies, understand overlap,    ║
║          and pick the right splitter for each document type  ║
╚══════════════════════════════════════════════════════════════╝
"""

from dotenv import load_dotenv
load_dotenv()

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_core.documents import Document

print("=" * 60)
print("09 — Chunking Strategies")
print("=" * 60)

LONG_TEXT = """
Introduction to Machine Learning

Machine learning is a branch of artificial intelligence that enables computers to learn from data.
Unlike traditional programming, where rules are explicitly coded, ML models discover patterns automatically.
There are three main types of machine learning: supervised, unsupervised, and reinforcement learning.

Supervised Learning

In supervised learning, the model is trained on labeled data.
Each training example has an input and a known correct output.
The model learns to map inputs to outputs by minimizing prediction error.
Common algorithms include linear regression, decision trees, and neural networks.
Applications include spam detection, image classification, and price prediction.

Unsupervised Learning

Unsupervised learning finds patterns in data without labels.
Clustering algorithms like K-means group similar data points together.
Dimensionality reduction techniques like PCA compress data while preserving structure.
These methods are used for customer segmentation and anomaly detection.

Reinforcement Learning

In reinforcement learning, an agent learns by interacting with an environment.
It receives rewards for good actions and penalties for bad ones.
Over time, the agent learns to maximize cumulative reward.
Famous applications include AlphaGo and game-playing AIs.
"""

# ─────────────────────────────────────────────────────────────
# SECTION 1: Effect of chunk_size
# Smaller chunks → more precise retrieval, less context per chunk
# Larger chunks → more context, but harder to find exact match
# ─────────────────────────────────────────────────────────────
print("\n📌 1. Effect of chunk_size")
print("─" * 40)

for chunk_size in [100, 300, 600]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=20)
    chunks   = splitter.split_text(LONG_TEXT)
    avg_len  = sum(len(c) for c in chunks) / len(chunks)
    print(f"  chunk_size={chunk_size:<4} → {len(chunks):2} chunks, avg {avg_len:.0f} chars each")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Effect of chunk_overlap
# Overlap creates "sliding window" — context at chunk boundaries
# isn't lost. Without overlap, a sentence split across chunks
# loses meaning in both.
# ─────────────────────────────────────────────────────────────
print("\n📌 2. Effect of chunk_overlap (at chunk boundary)")
print("─" * 40)

sample = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"

for overlap in [0, 5, 10]:
    s = CharacterTextSplitter(chunk_size=15, chunk_overlap=overlap, separator=" ")
    chunks = s.split_text(sample)
    print(f"  overlap={overlap:<3} → {len(chunks)} chunks: {chunks}")

print("  ↑ Notice: with overlap, boundary words appear in BOTH neighboring chunks")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Separator Priority in RecursiveCharacterTextSplitter
# Tries separators in order: \n\n → \n → ". " → " " → ""
# This preserves paragraph/sentence structure when possible.
# ─────────────────────────────────────────────────────────────
print("\n📌 3. Recursive Separator Priority")
print("─" * 40)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_text(LONG_TEXT)
print(f"  {len(chunks)} chunks produced. Each tries to end at a paragraph/sentence boundary:")
for i, chunk in enumerate(chunks[:4], 1):
    first_line = chunk.strip().split("\n")[0][:70]
    print(f"  [{i}] {first_line}...")

# ─────────────────────────────────────────────────────────────
# SECTION 4: TokenTextSplitter — split by actual tokens
# Use when you need precise control over token budget.
# Crucial for avoiding "context window exceeded" errors.
# ─────────────────────────────────────────────────────────────
print("\n📌 4. TokenTextSplitter — Count Real Tokens")
print("─" * 40)

try:
    token_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)
    token_chunks   = token_splitter.split_text(LONG_TEXT)
    print(f"  Token chunks: {len(token_chunks)}")
    print(f"  First chunk : {token_chunks[0][:100]}...")
    print(f"  ← Each chunk guaranteed ≤ 100 tokens (safe for embedding models)")
except Exception as e:
    print(f"  Note: install tiktoken for TokenTextSplitter: pip install tiktoken")

# ─────────────────────────────────────────────────────────────
# SECTION 5: Splitter Selection Guide
# ─────────────────────────────────────────────────────────────
print("\n📌 5. Which Splitter to Use?")
print("─" * 40)

guide = [
    ("RecursiveCharacterTextSplitter", "General text, articles, books (default choice)"),
    ("CharacterTextSplitter",          "When you want to split only on one separator"),
    ("TokenTextSplitter",              "When you have strict token limits (embedding models)"),
    ("MarkdownHeaderTextSplitter",     "Markdown docs — preserves header hierarchy in metadata"),
    ("HTMLHeaderTextSplitter",         "HTML pages — preserves heading structure"),
    ("SemanticChunker",                "Best quality splits (uses embeddings to find breaks)"),
    ("RecursiveJsonSplitter",          "JSON data — respects object/array boundaries"),
]

for name, use_case in guide:
    print(f"  {name:<38} → {use_case}")

# ─────────────────────────────────────────────────────────────
# SECTION 6: Metadata Preservation
# split_documents() carries metadata from the parent to each chunk.
# Always use this over split_text() when you have Document objects.
# ─────────────────────────────────────────────────────────────
print("\n📌 6. Metadata Preservation with split_documents()")
print("─" * 40)

parent_doc = Document(
    page_content=LONG_TEXT,
    metadata={"source": "ml_guide.txt", "author": "Anonymous", "topic": "ML"}
)

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks   = splitter.split_documents([parent_doc])

print(f"  Parent doc metadata : {parent_doc.metadata}")
print(f"  Chunks produced     : {len(chunks)}")
print(f"  Chunk[0] metadata   : {chunks[0].metadata}")
print(f"  ← Metadata is COPIED to every chunk automatically!")

# ─────────────────────────────────────────────────────────────
# ⚠️  COMMON MISTAKE 1: Using chunk_size without checking units
# RecursiveCharacterTextSplitter uses CHARACTER count by default.
# TokenTextSplitter uses TOKEN count.
# 1 token ≈ 4 characters → 1000 chars ≈ 250 tokens

# ⚠️  COMMON MISTAKE 2: chunk_overlap > chunk_size
# overlap must always be smaller than chunk_size or you get infinite chunks.

# ⚠️  COMMON MISTAKE 3: Splitting before loading metadata
# Split Documents (not raw strings) so metadata is carried through.
# ─────────────────────────────────────────────────────────────
print("\n⚠️  Splitting tips:")
print("   • 1000 chars ≈ 250 tokens (rule of thumb)")
print("   • chunk_overlap should be 10-20% of chunk_size")
print("   • Always split Documents, not raw strings (preserves metadata)")
print("   • For RAG: start with chunk_size=1000, chunk_overlap=200")
