"""
11 — Vector Stores
Example 01: Chroma and FAISS — CRUD operations, search types, metadata filtering
"""

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

DOCS = [
    Document(page_content="LangChain is a framework for building LLM applications.", metadata={"source": "langchain.txt", "topic": "langchain", "year": 2023}),
    Document(page_content="LCEL is LangChain Expression Language using the pipe operator.", metadata={"source": "lcel.txt", "topic": "langchain", "year": 2023}),
    Document(page_content="RAG stands for Retrieval-Augmented Generation.", metadata={"source": "rag.txt", "topic": "rag", "year": 2023}),
    Document(page_content="Chroma is a popular open-source vector database for local development.", metadata={"source": "chroma.txt", "topic": "vectorstore", "year": 2023}),
    Document(page_content="FAISS is Facebook AI Similarity Search — fast in-memory vector store.", metadata={"source": "faiss.txt", "topic": "vectorstore", "year": 2022}),
    Document(page_content="Pinecone is a managed cloud vector database for production use.", metadata={"source": "pinecone.txt", "topic": "vectorstore", "year": 2022}),
    Document(page_content="LangGraph enables stateful multi-agent graph orchestration.", metadata={"source": "langgraph.txt", "topic": "agents", "year": 2024}),
    Document(page_content="Agents use LLMs to decide which tools to call and when.", metadata={"source": "agents.txt", "topic": "agents", "year": 2023}),
]

# ─── 1. Chroma — In-Memory ────────────────────────────────────────────────────

print("=" * 55)
print("1. Chroma — In-Memory Vector Store")
print("=" * 55)

chroma_db = Chroma.from_documents(
    documents=DOCS,
    embedding=embeddings,
    collection_name="langchain_demo",
)

print(f"Collection size: {chroma_db._collection.count()} documents")

# Basic similarity search
results = chroma_db.similarity_search("What is LangChain?", k=3)
print("\nSimilarity search — 'What is LangChain?':")
for doc in results:
    print(f"  [{doc.metadata['source']}] {doc.page_content}")

# ─── 2. Similarity Search with Scores ────────────────────────────────────────

print("\n" + "=" * 55)
print("2. Similarity Search with Relevance Scores")
print("=" * 55)

results_with_scores = chroma_db.similarity_search_with_relevance_scores(
    "vector database for production", k=4
)
print("Query: 'vector database for production'")
for doc, score in results_with_scores:
    bar = "█" * int(score * 20)
    print(f"  {score:.3f} {bar:<20} | {doc.page_content[:55]}")

# ─── 3. MMR Search (Maximal Marginal Relevance) ───────────────────────────────

print("\n" + "=" * 55)
print("3. MMR Search — Reduces Redundancy")
print("=" * 55)

mmr_results = chroma_db.max_marginal_relevance_search(
    "vector store", k=3, fetch_k=6
)
print("MMR results for 'vector store' (diverse, less redundant):")
for doc in mmr_results:
    print(f"  [{doc.metadata['topic']}] {doc.page_content[:60]}")

# ─── 4. Metadata Filtering ────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("4. Metadata Filtering")
print("=" * 55)

# Filter by topic
agent_docs = chroma_db.similarity_search(
    "LLM orchestration",
    k=3,
    filter={"topic": "agents"},
)
print("Search with filter {'topic': 'agents'}:")
for doc in agent_docs:
    print(f"  {doc.page_content[:65]}")

# Filter by year
recent_docs = chroma_db.similarity_search(
    "LangChain tools",
    k=4,
    filter={"year": 2024},
)
print(f"\nFilter {{'year': 2024}} — found {len(recent_docs)} docs:")
for doc in recent_docs:
    print(f"  [{doc.metadata['year']}] {doc.page_content[:60]}")

# ─── 5. Add and Delete Documents ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("5. Adding and Deleting Documents")
print("=" * 55)

new_doc = Document(
    page_content="LangSmith is the observability and evaluation platform for LangChain.",
    metadata={"source": "langsmith.txt", "topic": "production", "year": 2024}
)
ids = chroma_db.add_documents([new_doc])
print(f"Added document with id: {ids[0]}")
print(f"New collection size: {chroma_db._collection.count()}")

chroma_db.delete(ids=ids)
print(f"Deleted document. Size restored to: {chroma_db._collection.count()}")

# ─── 6. Persist to Disk ───────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("6. Persisting Chroma to Disk")
print("=" * 55)

persistent_db = Chroma.from_documents(
    documents=DOCS,
    embedding=embeddings,
    persist_directory="./chroma_db_demo",
    collection_name="persistent_demo",
)
print("Saved to ./chroma_db_demo/")

# Reload from disk
reloaded_db = Chroma(
    persist_directory="./chroma_db_demo",
    embedding_function=embeddings,
    collection_name="persistent_demo",
)
print(f"Reloaded. Documents: {reloaded_db._collection.count()}")

# ─── 7. FAISS ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("7. FAISS — In-Memory Fast Search")
print("=" * 55)

faiss_db = FAISS.from_documents(DOCS, embeddings)

results = faiss_db.similarity_search("agent tools", k=3)
print("FAISS search for 'agent tools':")
for doc in results:
    print(f"  {doc.page_content[:65]}")

# Save and load FAISS
faiss_db.save_local("./faiss_index_demo")
reloaded_faiss = FAISS.load_local(
    "./faiss_index_demo",
    embeddings,
    allow_dangerous_deserialization=True,
)
print(f"\nFAISS reloaded. Index size: {reloaded_faiss.index.ntotal} vectors")

# ─── 8. As Retriever ──────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("8. Converting to Retriever")
print("=" * 55)

retriever = chroma_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 8},
)
docs = retriever.invoke("how to build agents in LangChain")
print("Retriever results:")
for doc in docs:
    print(f"  {doc.page_content[:65]}")

# Cleanup
import shutil
shutil.rmtree("./chroma_db_demo", ignore_errors=True)
shutil.rmtree("./faiss_index_demo", ignore_errors=True)
print("\nCleaned up demo directories.")
