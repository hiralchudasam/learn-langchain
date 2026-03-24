"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 08 — Document Loaders                                 ║
║  File:   01_document_loaders.py                              ║
║  Level:  Basic → Intermediate                                ║
║  Goal:   Load text from files, web, CSV, and directories.    ║
║          Understand the Document object and metadata.        ║
╚══════════════════════════════════════════════════════════════╝
"""

from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
from langchain_core.documents import Document

print("=" * 60)
print("08 — Document Loaders")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# SECTION 1: What is a Document?
# The universal container for loaded content in LangChain.
# page_content → the actual text
# metadata     → source info (filename, page, URL, etc.)
# ─────────────────────────────────────────────────────────────
print("\n📌 1. The Document Object")
print("─" * 40)

doc = Document(
    page_content="LangChain makes building LLM apps easy.",
    metadata={
        "source":  "langchain_guide.pdf",
        "page":    1,
        "author":  "Harrison Chase",
        "chapter": "Introduction",
    }
)

print(f"  Content  : {doc.page_content}")
print(f"  Source   : {doc.metadata['source']}")
print(f"  Page     : {doc.metadata['page']}")
print(f"  Keys     : {list(doc.metadata.keys())}")

# ─────────────────────────────────────────────────────────────
# SECTION 2: TextLoader — load plain text files
# ─────────────────────────────────────────────────────────────
print("\n📌 2. TextLoader")
print("─" * 40)

# Create a temp file to load
with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
    f.write("LangChain is a framework for LLM applications.\n")
    f.write("It supports chains, agents, and RAG.\n")
    f.write("LCEL is the modern way to compose chains.\n")
    tmpfile = f.name

from langchain_community.document_loaders import TextLoader

loader = TextLoader(tmpfile)
docs   = loader.load()

print(f"  Loaded   : {len(docs)} document(s)")
print(f"  Content  : {docs[0].page_content.strip()[:80]}...")
print(f"  Metadata : {docs[0].metadata}")
os.unlink(tmpfile)

# ─────────────────────────────────────────────────────────────
# SECTION 3: CSVLoader — load tabular data
# Each row becomes a separate Document.
# ─────────────────────────────────────────────────────────────
print("\n📌 3. CSVLoader — Each Row = One Document")
print("─" * 40)

import csv

with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "role", "experience"])
    writer.writeheader()
    writer.writerow({"name": "Rahul",  "role": "ML Engineer",     "experience": "3 years"})
    writer.writerow({"name": "Priya",  "role": "Data Scientist",   "experience": "5 years"})
    writer.writerow({"name": "Arjun",  "role": "Backend Dev",      "experience": "4 years"})
    csv_file = f.name

from langchain_community.document_loaders import CSVLoader

csv_loader = CSVLoader(csv_file)
csv_docs   = csv_loader.load()

print(f"  Rows loaded: {len(csv_docs)}")
for doc in csv_docs:
    print(f"  Row: {doc.page_content.strip()}")
    print(f"       Source: {doc.metadata}")
os.unlink(csv_file)

# ─────────────────────────────────────────────────────────────
# SECTION 4: WebBaseLoader — load from URLs
# Scrapes the visible text from web pages.
# ─────────────────────────────────────────────────────────────
print("\n📌 4. WebBaseLoader — Load from the Web")
print("─" * 40)

from langchain_community.document_loaders import WebBaseLoader

urls = [
    "https://python.langchain.com/docs/introduction/",
]

try:
    web_loader = WebBaseLoader(urls)
    web_docs   = web_loader.load()

    print(f"  Pages loaded  : {len(web_docs)}")
    print(f"  First page    : {web_docs[0].page_content[:200].strip()}...")
    print(f"  Metadata keys : {list(web_docs[0].metadata.keys())}")
except Exception as e:
    print(f"  ⏭️  Skipped (network error): {e}")

# ─────────────────────────────────────────────────────────────
# SECTION 5: DirectoryLoader — load all files in a folder
# Useful for ingesting an entire knowledge base.
# ─────────────────────────────────────────────────────────────
print("\n📌 5. DirectoryLoader — Load an Entire Folder")
print("─" * 40)

# Create a temp directory with multiple files
tmpdir = tempfile.mkdtemp()
files = {
    "intro.txt":    "LangChain is a framework for LLM applications.",
    "rag.txt":      "RAG stands for Retrieval-Augmented Generation.",
    "agents.txt":   "Agents use LLMs to decide which tools to call.",
}
for fname, content in files.items():
    with open(os.path.join(tmpdir, fname), "w") as f:
        f.write(content)

from langchain_community.document_loaders import DirectoryLoader

dir_loader = DirectoryLoader(
    tmpdir,
    glob="**/*.txt",         # load all .txt files recursively
    loader_cls=TextLoader,
    show_progress=False,
)
dir_docs = dir_loader.load()

print(f"  Files loaded: {len(dir_docs)}")
for doc in sorted(dir_docs, key=lambda d: d.metadata["source"]):
    fname = os.path.basename(doc.metadata["source"])
    print(f"  [{fname}] {doc.page_content[:60]}")

# Cleanup
import shutil
shutil.rmtree(tmpdir)

# ─────────────────────────────────────────────────────────────
# SECTION 6: Lazy Loading — memory-efficient for large corpora
# Use lazy_load() to process one document at a time.
# ─────────────────────────────────────────────────────────────
print("\n📌 6. Lazy Loading — Memory-Efficient")
print("─" * 40)

# Create sample files again
tmpdir2 = tempfile.mkdtemp()
for i in range(5):
    with open(os.path.join(tmpdir2, f"doc_{i}.txt"), "w") as f:
        f.write(f"Document {i}: This is content for document number {i}.")

lazy_loader = DirectoryLoader(tmpdir2, glob="*.txt", loader_cls=TextLoader)

# lazy_load() is a generator — doesn't load everything into memory at once
total_words = 0
for doc in lazy_loader.lazy_load():
    total_words += len(doc.page_content.split())
    # Process one doc at a time — efficient for millions of docs

print(f"  Processed docs via lazy_load()")
print(f"  Total word count: {total_words}")
print(f"  ← No RAM spike — each doc processed and discarded")

shutil.rmtree(tmpdir2)

# ─────────────────────────────────────────────────────────────
# ⚠️  COMMON MISTAKE 1: Loading huge files with .load()
# .load() puts everything in memory at once.
# For large datasets use .lazy_load() instead.

# ⚠️  COMMON MISTAKE 2: Not checking metadata after loading
# Different loaders add different metadata keys.
# Always print doc.metadata to know what you have.

# ⚠️  COMMON MISTAKE 3: Forgetting encoding for non-English text
# TextLoader("file.txt", encoding="utf-8") for Hindi/Chinese/etc.
# ─────────────────────────────────────────────────────────────
print("\n⚠️  Document loader tips:")
print("   • Use lazy_load() for large corpora (memory-efficient)")
print("   • Print doc.metadata to see what each loader provides")
print("   • TextLoader needs encoding='utf-8' for non-ASCII text")
print("   • WebBaseLoader respects robots.txt — some sites may block")
