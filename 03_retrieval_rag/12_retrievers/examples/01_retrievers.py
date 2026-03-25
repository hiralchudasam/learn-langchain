"""
12 — Retrievers
Example 01: VectorStore, MultiQuery, ContextualCompression, Ensemble retrievers
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
import logging

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

DOCS = [
    Document(page_content="LangChain is a Python framework for building LLM applications. It supports chains, agents, memory, and retrieval.", metadata={"source": "intro.txt"}),
    Document(page_content="LCEL (LangChain Expression Language) lets you compose chains using the pipe operator. Every component is a Runnable.", metadata={"source": "lcel.txt"}),
    Document(page_content="RAG (Retrieval-Augmented Generation) retrieves relevant docs before passing them to an LLM for grounded generation.", metadata={"source": "rag.txt"}),
    Document(page_content="LangGraph builds stateful multi-agent graphs with cycles, human-in-the-loop, and checkpointing support.", metadata={"source": "langgraph.txt"}),
    Document(page_content="Chroma is an open-source embedding database. It runs locally and persists to disk. Great for development.", metadata={"source": "chroma.txt"}),
    Document(page_content="FAISS is a fast similarity search library from Facebook AI. It operates fully in memory.", metadata={"source": "faiss.txt"}),
    Document(page_content="LangSmith provides tracing, debugging, and evaluation for LangChain apps. It auto-traces all runs.", metadata={"source": "langsmith.txt"}),
    Document(page_content="Agents use LLMs as reasoning engines. The LLM decides which tools to call and when to stop.", metadata={"source": "agents.txt"}),
    Document(page_content="Tools are functions that agents can call. The @tool decorator is the simplest way to define a tool.", metadata={"source": "tools.txt"}),
    Document(page_content="Embeddings are numerical vectors representing the semantic meaning of text. Similar texts have similar vectors.", metadata={"source": "embeddings.txt"}),
]

vectorstore = Chroma.from_documents(DOCS, embeddings, collection_name="retriever_demo")

# ─── 1. Basic VectorStoreRetriever ────────────────────────────────────────────

print("=" * 55)
print("1. VectorStoreRetriever (similarity)")
print("=" * 55)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("how do I build an agent?")
print("Query: 'how do I build an agent?'")
for doc in docs:
    print(f"  [{doc.metadata['source']}] {doc.page_content[:70]}")

# ─── 2. MMR Retriever (less redundant results) ───────────────────────────────

print("\n" + "=" * 55)
print("2. MMR Retriever (diversity)")
print("=" * 55)

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 8, "lambda_mult": 0.5},
)
docs = mmr_retriever.invoke("vector store and similarity search")
print("Query: 'vector store and similarity search' (MMR — diverse results):")
for doc in docs:
    print(f"  [{doc.metadata['source']}] {doc.page_content[:70]}")

# ─── 3. MultiQueryRetriever ───────────────────────────────────────────────────

print("\n" + "=" * 55)
print("3. MultiQueryRetriever (generates query variants)")
print("=" * 55)

logging.basicConfig(level=logging.WARNING)  # suppress verbose logging
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    llm=llm,
)

docs = multi_retriever.invoke("tell me about LangChain tools")
print(f"Query: 'tell me about LangChain tools'")
print(f"Retrieved {len(docs)} unique docs via MultiQuery:")
for doc in docs:
    print(f"  [{doc.metadata['source']}] {doc.page_content[:65]}")

# ─── 4. ContextualCompressionRetriever ────────────────────────────────────────

print("\n" + "=" * 55)
print("4. ContextualCompressionRetriever")
print("=" * 55)

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
)

docs = compression_retriever.invoke("What is LCEL and why use it?")
print("Query: 'What is LCEL and why use it?'")
print("Compressed results (only relevant sentences kept):")
for doc in docs:
    print(f"  [{doc.metadata['source']}] {doc.page_content}")

# ─── 5. EnsembleRetriever (Hybrid: BM25 + Semantic) ──────────────────────────

print("\n" + "=" * 55)
print("5. EnsembleRetriever (BM25 + Semantic = Hybrid Search)")
print("=" * 55)

bm25_retriever = BM25Retriever.from_documents(DOCS)
bm25_retriever.k = 3

semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.4, 0.6],
)

docs = ensemble_retriever.invoke("LangChain LCEL pipe operator")
print("Query: 'LangChain LCEL pipe operator' (hybrid BM25+semantic):")
for doc in docs:
    print(f"  [{doc.metadata['source']}] {doc.page_content[:70]}")

# ─── 6. Retriever Comparison ──────────────────────────────────────────────────

print("\n" + "=" * 55)
print("6. Retriever Comparison")
print("=" * 55)

query = "how to observe and debug my chains"
print(f"Query: '{query}'\n")

retrievers = {
    "VectorStore (k=3)": vectorstore.as_retriever(search_kwargs={"k": 3}),
    "MMR (k=3)":         vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3}),
    "BM25 (k=3)":        bm25_retriever,
}

for name, ret in retrievers.items():
    docs = ret.invoke(query)
    sources = [d.metadata["source"] for d in docs]
    print(f"  {name}: {sources}")
