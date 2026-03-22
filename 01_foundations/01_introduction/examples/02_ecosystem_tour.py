"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 01 — Introduction to LangChain                       ║
║  File:   02_ecosystem_tour.py                                ║
║  Level:  Intermediate                                        ║
║  Goal:   Understand every component in the LangChain         ║
║          ecosystem and how they fit together                 ║
╚══════════════════════════════════════════════════════════════╝
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

print("=" * 60)
print("01 — LangChain Ecosystem Tour")
print("=" * 60)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ─────────────────────────────────────────────────────────────
# COMPONENT 1: langchain-core
# The foundation — defines all base interfaces.
# Every other package builds on top of this.
# Key classes: Runnable, BaseMessage, Document, PromptTemplate
# ─────────────────────────────────────────────────────────────
print("\n📦 1. langchain-core — Base Interfaces")
print("─" * 40)

# Document: the universal container for text + metadata
doc = Document(
    page_content="LangChain makes building LLM apps easy.",
    metadata={"source": "docs", "page": 1, "author": "Harrison Chase"}
)
print(f"Document content : {doc.page_content}")
print(f"Document metadata: {doc.metadata}")

# Messages: the building blocks of chat
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is LangChain?"),
]
print(f"\nMessage types    : {[type(m).__name__ for m in messages]}")

# ─────────────────────────────────────────────────────────────
# COMPONENT 2: ChatModel (langchain-openai)
# Sends messages to an LLM and returns an AIMessage.
# Every provider (OpenAI, Anthropic, etc.) has the SAME interface.
# ─────────────────────────────────────────────────────────────
print("\n📦 2. ChatModel — Talk to LLMs")
print("─" * 40)

response = llm.invoke(messages)
print(f"Response type    : {type(response).__name__}")
print(f"Response content : {response.content}")
print(f"Token usage      : {response.usage_metadata}")

# ─────────────────────────────────────────────────────────────
# COMPONENT 3: Embeddings (langchain-openai)
# Converts text into a vector of numbers.
# Used for semantic search — similar texts → similar vectors.
# ─────────────────────────────────────────────────────────────
print("\n📦 3. Embeddings — Turn Text into Vectors")
print("─" * 40)

embedder = OpenAIEmbeddings(model="text-embedding-3-small")
vector = embedder.embed_query("What is LangChain?")
print(f"Input text       : 'What is LangChain?'")
print(f"Vector length    : {len(vector)} dimensions")
print(f"First 5 values   : {[round(v, 4) for v in vector[:5]]}")
print(f"(Each number encodes a dimension of meaning)")

# ─────────────────────────────────────────────────────────────
# COMPONENT 4: LCEL Runnables
# Every LangChain object is a Runnable.
# They all share the same interface: .invoke(), .batch(), .stream()
# Chain them together with the | pipe operator.
# ─────────────────────────────────────────────────────────────
print("\n📦 4. LCEL — Composing with the Pipe Operator")
print("─" * 40)

chain = (
    ChatPromptTemplate.from_template("Name 3 uses of {technology}. Be brief.")
    | llm
    | StrOutputParser()
)

# .invoke() — single input
result = chain.invoke({"technology": "LangChain"})
print(f".invoke() result :\n{result}")

# .batch() — multiple inputs in parallel
results = chain.batch([
    {"technology": "RAG"},
    {"technology": "LangGraph"},
])
print(f"\n.batch() results :")
for i, r in enumerate(results, 1):
    print(f"  [{i}] {r[:80]}...")

# ─────────────────────────────────────────────────────────────
# COMPONENT 5: Runnable Inspection
# Every chain/runnable can tell you its input/output schema.
# Useful for debugging what a chain expects.
# ─────────────────────────────────────────────────────────────
print("\n📦 5. Runnable Inspection — Understand Your Chain")
print("─" * 40)

print(f"Input schema  : {chain.input_schema.model_json_schema()}")
print(f"Output schema : {chain.output_schema.model_json_schema()}")

# ─────────────────────────────────────────────────────────────
# COMPONENT 6: The Ecosystem Map
# ─────────────────────────────────────────────────────────────
print("\n📦 6. The Full Ecosystem at a Glance")
print("─" * 40)

ecosystem = {
    "langchain-core":      "Base interfaces — Runnable, Document, BaseMessage",
    "langchain":           "Chains, agents, memory, retrievers",
    "langchain-community": "100+ integrations (loaders, tools, vectorstores)",
    "langchain-openai":    "OpenAI LLMs + embeddings",
    "langchain-anthropic": "Anthropic Claude LLMs",
    "langgraph":           "Stateful multi-agent graph orchestration",
    "langserve":           "Deploy chains as REST APIs",
    "langsmith":           "Tracing, evaluation, observability",
}

for pkg, desc in ecosystem.items():
    print(f"  {pkg:<25} → {desc}")

# ─────────────────────────────────────────────────────────────
# ⚠️  COMMON MISTAKE: Mixing old and new import paths
# OLD (v0.1): from langchain.chat_models import ChatOpenAI  ❌
# NEW (v0.3): from langchain_openai import ChatOpenAI       ✅
#
# OLD (v0.1): from langchain.schema import HumanMessage     ❌
# NEW (v0.3): from langchain_core.messages import HumanMessage ✅
# ─────────────────────────────────────────────────────────────
print("\n⚠️  Remember: Always use the NEW import paths (v0.3+)")
print("   from langchain_openai import ChatOpenAI       ✅")
print("   from langchain.chat_models import ChatOpenAI  ❌ (deprecated)")
