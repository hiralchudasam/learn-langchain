"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 11 — Vector Stores                                    ║
║  File:   02_vector_store_patterns.py                         ║
║  Level:  Advanced                                            ║
║  Goal:   Multi-collection, update/delete, score thresholds,  ║
║          and production patterns                             ║
╚══════════════════════════════════════════════════════════════╝
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import shutil

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
print("=" * 60)
print("11 — Vector Store Advanced Patterns")
print("=" * 60)

DOCS = [
    Document(page_content="Python is a versatile programming language used in AI.",  metadata={"topic": "python", "level": "beginner"}),
    Document(page_content="Python decorators wrap functions to add behavior.",       metadata={"topic": "python", "level": "advanced"}),
    Document(page_content="LangChain is a framework for LLM applications.",         metadata={"topic": "langchain", "level": "beginner"}),
    Document(page_content="LCEL uses the pipe operator to compose chains.",          metadata={"topic": "langchain", "level": "intermediate"}),
    Document(page_content="RAG retrieves documents before generating answers.",     metadata={"topic": "rag", "level": "intermediate"}),
    Document(page_content="Vector stores use cosine similarity for search.",        metadata={"topic": "rag", "level": "advanced"}),
]

db = Chroma.from_documents(DOCS, embeddings, collection_name="patterns_demo")

# 1. Score threshold — only return highly relevant results
print("\n📌 1. Score Threshold Retrieval")
print("─" * 40)
threshold_retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.75, "k": 5},
)
results = threshold_retriever.invoke("Python programming language")
print(f"  Query: 'Python programming language'")
print(f"  Results above 0.75 threshold: {len(results)}")
for r in results:
    print(f"    [{r.metadata['topic']}] {r.page_content[:60]}")

# 2. Filter by metadata
print("\n📌 2. Metadata Filtering")
print("─" * 40)
for level in ["beginner", "advanced"]:
    results = db.similarity_search("LangChain Python", k=3, filter={"level": level})
    print(f"  [{level}] found {len(results)} docs: {[d.page_content[:40] for d in results]}")

# 3. Incremental updates
print("\n📌 3. Incremental Updates (add/delete)")
print("─" * 40)
print(f"  Before: {db._collection.count()} docs")

new_docs = [
    Document(page_content="LangGraph builds stateful multi-agent graphs.", metadata={"topic": "langgraph", "level": "advanced"}),
]
ids = db.add_documents(new_docs)
print(f"  Added 1 doc. Now: {db._collection.count()} docs")

db.delete(ids=ids)
print(f"  Deleted 1 doc. Back to: {db._collection.count()} docs")

# 4. Similarity search with scores
print("\n📌 4. Scores — See How Relevant Each Result Is")
print("─" * 40)
results = db.similarity_search_with_relevance_scores("LLM applications framework", k=4)
for doc, score in sorted(results, reverse=True, key=lambda x: x[1]):
    bar = "█" * int(score * 20)
    print(f"  {score:.3f} {bar:<20} {doc.page_content[:50]}")

# ⚠️ COMMON MISTAKES
print("\n⚠️  Vector store tips:")
print("   • Use score_threshold to filter out irrelevant results")
print("   • Always filter by metadata when you have structured data")
print("   • persist_directory is essential — don't re-embed every run")
