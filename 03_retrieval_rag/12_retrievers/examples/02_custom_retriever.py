"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 12 — Retrievers                                       ║
║  File:   02_custom_retriever.py                              ║
║  Level:  Advanced                                            ║
║  Goal:   Build a custom retriever, time-weighted retrieval,  ║
║          and combine retrievers intelligently                ║
╚══════════════════════════════════════════════════════════════╝
"""

from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.vectorstores import Chroma
from pydantic import Field
from typing import List

llm        = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print("=" * 60)
print("12 — Custom Retrievers")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# CUSTOM RETRIEVER: Build your own retriever from scratch.
# Extend BaseRetriever and implement _get_relevant_documents().
# ─────────────────────────────────────────────────────────────
print("\n📌 1. Custom Retriever from Scratch")
print("─" * 40)

class KeywordRetriever(BaseRetriever):
    """Simple keyword-matching retriever — no embeddings needed."""
    documents: List[Document] = Field(default_factory=list)
    k: int = 3

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        query_words = set(query.lower().split())
        scored = []
        for doc in self.documents:
            doc_words = set(doc.page_content.lower().split())
            overlap   = len(query_words & doc_words)
            if overlap > 0:
                scored.append((overlap, doc))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in scored[:self.k]]

docs = [
    Document(page_content="LangChain is a Python framework for LLM apps."),
    Document(page_content="RAG combines retrieval with LLM generation."),
    Document(page_content="Python is a popular programming language."),
    Document(page_content="LangGraph builds multi-agent graphs in Python."),
]

retriever = KeywordRetriever(documents=docs, k=2)
results   = retriever.invoke("Python LangChain framework")
print(f"  Query: 'Python LangChain framework'")
for r in results:
    print(f"  → {r.page_content}")

# ─────────────────────────────────────────────────────────────
# TIME-WEIGHTED RETRIEVAL: Recent docs get a boost
# ─────────────────────────────────────────────────────────────
print("\n📌 2. Time-Weighted Retrieval (recent = more relevant)")
print("─" * 40)

from langchain.retrievers import TimeWeightedVectorStoreRetriever

now = datetime.now()
dated_docs = [
    Document(page_content="LangChain v0.1 introduced chains.",     metadata={"last_accessed_at": now - timedelta(days=90)}),
    Document(page_content="LangChain v0.2 introduced LCEL.",       metadata={"last_accessed_at": now - timedelta(days=30)}),
    Document(page_content="LangChain v0.3 is the latest release.", metadata={"last_accessed_at": now - timedelta(days=1)}),
]

vstore    = Chroma(embedding_function=embeddings, collection_name="time_weighted")
vstore.add_documents(dated_docs)

time_retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vstore,
    decay_rate=0.01,  # lower = slower decay (older docs still relevant longer)
    k=2,
)
results = time_retriever.invoke("LangChain release")
print(f"  Time-weighted results (recent preferred):")
for r in results:
    print(f"  → {r.page_content}")

print("\n⚠️  Retriever tips:")
print("   • Build custom retrievers for keyword, SQL, or API-based search")
print("   • Use TimeWeightedRetriever for news/events where recency matters")
print("   • All retrievers share the same .invoke() interface — composable!")
