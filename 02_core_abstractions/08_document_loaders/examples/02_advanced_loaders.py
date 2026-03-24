"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 08 — Document Loaders                                 ║
║  File:   02_advanced_loaders.py                              ║
║  Level:  Advanced                                            ║
║  Goal:   PDF loading with metadata, web scraping, custom     ║
║          loaders, and document preprocessing pipeline        ║
╚══════════════════════════════════════════════════════════════╝
"""

from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from typing import Iterator

print("=" * 60)
print("08 — Advanced Document Loaders")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# SECTION 1: JSONLoader — extract specific fields from JSON
# Great for loading API responses, structured data files.
# ─────────────────────────────────────────────────────────────
print("\n📌 1. JSONLoader — Extract Fields from JSON")
print("─" * 40)

import json

sample_json = [
    {"id": 1, "title": "Introduction to LangChain", "content": "LangChain is a framework for LLM apps.", "author": "Alice"},
    {"id": 2, "title": "RAG Explained",             "content": "RAG retrieves docs before generating.", "author": "Bob"},
    {"id": 3, "title": "LangGraph Guide",            "content": "LangGraph builds stateful agents.",    "author": "Alice"},
]

with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump(sample_json, f)
    json_file = f.name

try:
    from langchain_community.document_loaders import JSONLoader

    loader = JSONLoader(
        file_path=json_file,
        jq_schema=".[].content",            # extract the 'content' field from each item
        text_content=True,
        metadata_func=lambda rec, meta: {   # add custom metadata from the record
            **meta,
            "title":  rec.get("title"),
            "author": rec.get("author"),
            "id":     rec.get("id"),
        },
    )
    docs = loader.load()
    print(f"  Loaded {len(docs)} JSON records")
    for doc in docs:
        print(f"  [{doc.metadata.get('title')}] by {doc.metadata.get('author')}")
        print(f"    Content: {doc.page_content[:60]}")
except ImportError:
    print("  Install jq: pip install jq")
finally:
    os.unlink(json_file)

# ─────────────────────────────────────────────────────────────
# SECTION 2: Building a Custom Document Loader
# Extend BaseLoader when no built-in loader fits your source.
# ─────────────────────────────────────────────────────────────
print("\n📌 2. Custom Loader — Database / API Source")
print("─" * 40)

class InMemoryDBLoader(BaseLoader):
    """
    Custom loader that reads from an in-memory 'database'.
    In production, replace with real DB/API calls.
    """

    FAKE_DB = [
        {"id": 1, "question": "What is LangChain?",    "answer": "A framework for LLM apps.", "category": "langchain"},
        {"id": 2, "question": "What is RAG?",           "answer": "Retrieval-Augmented Generation.", "category": "rag"},
        {"id": 3, "question": "What is LangGraph?",    "answer": "A stateful agent graph library.", "category": "langchain"},
        {"id": 4, "question": "What are embeddings?",   "answer": "Numerical vectors representing text meaning.", "category": "rag"},
    ]

    def __init__(self, category: str = None):
        self.category = category  # optional filter

    def lazy_load(self) -> Iterator[Document]:
        """Yield one Document at a time — memory efficient."""
        for row in self.FAKE_DB:
            if self.category and row["category"] != self.category:
                continue
            # Format Q&A as document content
            content = f"Q: {row['question']}\nA: {row['answer']}"
            yield Document(
                page_content=content,
                metadata={
                    "source":   "in_memory_db",
                    "id":       row["id"],
                    "category": row["category"],
                }
            )

    def load(self) -> list[Document]:
        return list(self.lazy_load())

# Load all docs
all_docs = InMemoryDBLoader().load()
print(f"  All docs: {len(all_docs)}")

# Load filtered docs
rag_docs = InMemoryDBLoader(category="rag").load()
print(f"  RAG-only docs: {len(rag_docs)}")
for doc in rag_docs:
    print(f"    [{doc.metadata['id']}] {doc.page_content[:60]}")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Document Preprocessing Pipeline
# Clean, normalize, and enrich documents after loading.
# ─────────────────────────────────────────────────────────────
print("\n📌 3. Document Preprocessing Pipeline")
print("─" * 40)

def remove_extra_whitespace(doc: Document) -> Document:
    """Collapse multiple spaces/newlines into single ones."""
    cleaned = " ".join(doc.page_content.split())
    return Document(page_content=cleaned, metadata=doc.metadata)

def add_word_count(doc: Document) -> Document:
    """Add word count to metadata."""
    count = len(doc.page_content.split())
    return Document(
        page_content=doc.page_content,
        metadata={**doc.metadata, "word_count": count},
    )

def filter_short_docs(docs: list[Document], min_words: int = 5) -> list[Document]:
    """Remove documents that are too short to be useful."""
    return [d for d in docs if d.metadata.get("word_count", 0) >= min_words]

def add_source_tag(doc: Document, tag: str) -> Document:
    """Tag every document with a processing label."""
    return Document(
        page_content=doc.page_content,
        metadata={**doc.metadata, "processed_by": tag},
    )

# Run the pipeline
raw_docs = InMemoryDBLoader().load()
print(f"  Raw docs       : {len(raw_docs)}")

processed = raw_docs
processed = [remove_extra_whitespace(d) for d in processed]
processed = [add_word_count(d) for d in processed]
processed = filter_short_docs(processed, min_words=3)
processed = [add_source_tag(d, "v1-pipeline") for d in processed]

print(f"  After pipeline : {len(processed)}")
for doc in processed[:2]:
    print(f"  Metadata: {doc.metadata}")
    print(f"  Content : {doc.page_content[:60]}")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Multi-source loading with metadata tagging
# Load from multiple sources and tag each with its origin.
# ─────────────────────────────────────────────────────────────
print("\n📌 4. Multi-Source Loading")
print("─" * 40)

import shutil
from langchain_community.document_loaders import TextLoader, DirectoryLoader

tmpdir = tempfile.mkdtemp()

# Create files from different "sources"
sources = {
    "langchain_docs.txt":  "LangChain is a framework for building LLM apps.\nIt supports chains and agents.",
    "rag_guide.txt":       "RAG retrieves relevant documents before generation.\nIt reduces hallucination.",
    "agents_overview.txt": "Agents use LLMs to decide which tools to call.\nReAct is a popular pattern.",
}

for fname, content in sources.items():
    with open(os.path.join(tmpdir, fname), "w") as f:
        f.write(content)

loader = DirectoryLoader(tmpdir, glob="*.txt", loader_cls=TextLoader)
all_loaded = loader.load()

print(f"  Loaded {len(all_loaded)} files:")
for doc in all_loaded:
    fname = os.path.basename(doc.metadata["source"])
    words = len(doc.page_content.split())
    print(f"  [{fname:<25}] {words} words | {doc.page_content[:50]}...")

shutil.rmtree(tmpdir)

# ─────────────────────────────────────────────────────────────
# ⚠️  COMMON MISTAKE 1: Loading binary files as TextLoader
# TextLoader is for plain text only.
# Use PyPDFLoader for PDFs, UnstructuredLoader for Word/HTML.

# ⚠️  COMMON MISTAKE 2: Not adding metadata
# Source information (file, page, URL) is critical for citations.
# Always set meaningful metadata when building custom loaders.

# ⚠️  COMMON MISTAKE 3: Skipping preprocessing
# Raw loaded text often has excessive whitespace, headers, footers.
# Always run a cleaning step before splitting and embedding.
# ─────────────────────────────────────────────────────────────
print("\n⚠️  Document loader tips:")
print("   • Always add metadata — you'll need it for citations")
print("   • Build custom loaders for APIs, databases, and proprietary sources")
print("   • Clean documents (whitespace, noise) before embedding")
print("   • Use lazy_load() for large corpora to avoid memory issues")
