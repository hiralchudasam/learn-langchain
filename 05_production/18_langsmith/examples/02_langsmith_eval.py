"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 18 — LangSmith                                        ║
║  File:   02_langsmith_eval.py                                ║
║  Level:  Advanced                                            ║
║  Goal:   Datasets, evaluation runs, custom evaluators,       ║
║          A/B testing two chains with LangSmith               ║
╚══════════════════════════════════════════════════════════════╝
"""

from dotenv import load_dotenv
load_dotenv()

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable, Client

llm_mini = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_4o   = ChatOpenAI(model="gpt-4o",      temperature=0)

print("=" * 60)
print("18 — LangSmith Evaluation")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# SECTION 1: @traceable — trace any Python function
# Not just LangChain components — any function can be traced.
# ─────────────────────────────────────────────────────────────
print("\n📌 1. @traceable — Trace Custom Functions")
print("─" * 40)

@traceable(name="preprocess_input")
def preprocess(text: str) -> str:
    """Clean and normalize user input before sending to LLM."""
    return text.strip().lower().replace("  ", " ")

@traceable(name="postprocess_output")
def postprocess(text: str) -> str:
    """Format model output for display."""
    return text.strip().capitalize()

@traceable(name="qa_pipeline")
def run_qa(question: str) -> str:
    clean  = preprocess(question)
    chain  = (
        ChatPromptTemplate.from_template("Answer briefly: {q}")
        | llm_mini | StrOutputParser()
    )
    raw    = chain.invoke({"q": clean})
    return postprocess(raw)

result = run_qa("  WHAT IS LANGCHAIN?  ")
print(f"  Result: {result[:100]}")
print(f"  → Each function (preprocess, postprocess, qa_pipeline) is traced")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Adding tags and metadata to runs
# Tags help filter runs in the LangSmith UI.
# Metadata adds structured info (user_id, version, env, etc.)
# ─────────────────────────────────────────────────────────────
print("\n📌 2. Tags and Metadata on Runs")
print("─" * 40)

chain = (
    ChatPromptTemplate.from_template("Summarize {topic} in one sentence.")
    | llm_mini | StrOutputParser()
)

# Tag and label this specific run
result = chain.invoke(
    {"topic": "LangSmith"},
    config={
        "tags":       ["production", "summarizer", "v2"],
        "metadata":   {"user_id": "user_42", "env": "prod", "version": "2.1.0"},
        "run_name":   "summarizer-prod-run",
    },
)
print(f"  Result: {result[:80]}")
print(f"  → Run tagged with: production, summarizer, v2")
print(f"  → Metadata: user_id=user_42, env=prod, version=2.1.0")
print(f"  → Find in LangSmith by filtering on these tags")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Comparing two chains (A/B test)
# Run the same questions through two different chains.
# Compare quality, speed, and cost side by side.
# ─────────────────────────────────────────────────────────────
print("\n📌 3. A/B Testing Two Chains")
print("─" * 40)

chain_a = (  # gpt-4o-mini (fast, cheap)
    ChatPromptTemplate.from_template("Answer concisely: {question}")
    | llm_mini | StrOutputParser()
)
chain_b = (  # gpt-4o (slower, expensive, higher quality)
    ChatPromptTemplate.from_template("Answer thoroughly: {question}")
    | llm_4o | StrOutputParser()
)

from langchain_community.callbacks import get_openai_callback
import time

questions = [
    "What is LangChain?",
    "How does RAG work?",
    "What is LangGraph for?",
]

print(f"  {'Question':<30} {'Chain A tokens':<16} {'Chain B tokens':<16}")
print(f"  {'─'*30} {'─'*16} {'─'*16}")

for q in questions:
    with get_openai_callback() as cb_a:
        ans_a = chain_a.invoke({"question": q})

    with get_openai_callback() as cb_b:
        ans_b = chain_b.invoke({"question": q})

    print(f"  {q[:28]:<30} {cb_a.total_tokens:<16} {cb_b.total_tokens:<16}")

print(f"\n  → In LangSmith: tag chains 'model:gpt-4o-mini' and 'model:gpt-4o'")
print(f"  → Compare latency, cost, and quality side by side in the UI")

# ─────────────────────────────────────────────────────────────
# SECTION 4: LangSmith Client (if API key set)
# ─────────────────────────────────────────────────────────────
print("\n📌 4. LangSmith Client — Programmatic Access")
print("─" * 40)

if os.getenv("LANGCHAIN_API_KEY"):
    try:
        client = Client()
        projects = list(client.list_projects())
        print(f"  Connected to LangSmith!")
        print(f"  Projects: {[p.name for p in projects[:3]]}")
    except Exception as e:
        print(f"  LangSmith client error: {e}")
else:
    print("  Set LANGCHAIN_API_KEY in .env to enable LangSmith")
    print("  Without it: code works, but traces won't be sent")

print("\n⚠️  LangSmith tips:")
print("   • LANGCHAIN_TRACING_V2=true enables auto-tracing (no code changes)")
print("   • Use tags to group runs by feature/experiment/version")
print("   • Use run_name for easy searching in the LangSmith UI")
print("   • A/B test by tagging two chains differently, compare in UI")
