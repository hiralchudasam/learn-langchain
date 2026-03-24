"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 05 — LCEL (LangChain Expression Language)            ║
║  File:   02_lcel_advanced.py                                 ║
║  Level:  Advanced                                            ║
║  Goal:   Advanced LCEL patterns — configurable runnables,    ║
║          dynamic routing, async, and chain inspection        ║
╚══════════════════════════════════════════════════════════════╝
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
    RunnableBranch,
    RunnableConfig,
    ConfigurableField,
)

llm   = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

print("=" * 60)
print("05 — LCEL Advanced Patterns")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# SECTION 1: RunnableLambda — wrap any Python function
# Lets you insert arbitrary Python logic into a chain.
# The function must accept ONE argument (the chain's current value).
# ─────────────────────────────────────────────────────────────
print("\n📌 1. RunnableLambda — Inject Python Logic")
print("─" * 40)

def word_count(text: str) -> dict:
    """Adds word count metadata to the output."""
    return {"text": text, "word_count": len(text.split()), "char_count": len(text)}

def format_report(data: dict) -> str:
    return (f"{data['text']}\n\n"
            f"[Words: {data['word_count']} | Chars: {data['char_count']}]")

chain = (
    ChatPromptTemplate.from_template("Summarize {topic} in 3 sentences.")
    | llm
    | parser
    | RunnableLambda(word_count)
    | RunnableLambda(format_report)
)

result = chain.invoke({"topic": "neural networks"})
print(result)

# ─────────────────────────────────────────────────────────────
# SECTION 2: RunnablePassthrough — pass input alongside output
# Used when you need BOTH the original input AND a transformed version.
# ─────────────────────────────────────────────────────────────
print("\n📌 2. RunnablePassthrough — Keep Original Input")
print("─" * 40)

# Pattern: generate answer AND keep the original question
qa_chain = RunnableParallel(
    question = RunnablePassthrough(),                         # keeps original
    answer   = (
        ChatPromptTemplate.from_template("Answer in one sentence: {question}")
        | llm
        | parser
    ),
    sources  = RunnableLambda(lambda x: ["wikipedia", "docs"]),  # mock sources
)

result = qa_chain.invoke({"question": "What is a transformer model?"})
print(f"  Question : {result['question']['question']}")
print(f"  Answer   : {result['answer']}")
print(f"  Sources  : {result['sources']}")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Nested parallel chains
# Run sub-chains in parallel, combine results, then process further.
# ─────────────────────────────────────────────────────────────
print("\n📌 3. Nested Parallel → Sequential")
print("─" * 40)

def make_mini_chain(instruction: str):
    return (
        ChatPromptTemplate.from_template(f"{instruction}: {{topic}}")
        | llm | parser
    )

# Step 1: gather in parallel
gather = RunnableParallel(
    pros      = make_mini_chain("List 2 pros of"),
    cons      = make_mini_chain("List 2 cons of"),
    summary   = make_mini_chain("Write one sentence about"),
)

# Step 2: combine into a final report
def combine(data: dict) -> str:
    return (f"SUMMARY: {data['summary']}\n\n"
            f"PROS:\n{data['pros']}\n\n"
            f"CONS:\n{data['cons']}")

report_chain = (
    gather
    | RunnableLambda(combine)
)

result = report_chain.invoke({"topic": "Python"})
print(result[:300])

# ─────────────────────────────────────────────────────────────
# SECTION 4: Configurable Runnables — swap parts at runtime
# Change model, temperature, or any parameter without rewriting the chain.
# ─────────────────────────────────────────────────────────────
print("\n📌 4. Configurable Runnables")
print("─" * 40)

# Make model and temperature swappable at invoke time
configurable_llm = ChatOpenAI(model="gpt-4o-mini").configurable_fields(
    model_name=ConfigurableField(
        id="model",
        name="LLM Model",
        description="The model to use",
    ),
    temperature=ConfigurableField(
        id="temperature",
        name="Temperature",
        description="Creativity level",
    ),
)

config_chain = (
    ChatPromptTemplate.from_template("Write a tagline for: {product}")
    | configurable_llm
    | parser
)

# Different configs, same chain
tagline_creative = config_chain.invoke(
    {"product": "AI coding assistant"},
    config={"configurable": {"model": "gpt-4o-mini", "temperature": 1.2}},
)
tagline_factual = config_chain.invoke(
    {"product": "AI coding assistant"},
    config={"configurable": {"model": "gpt-4o-mini", "temperature": 0.0}},
)

print(f"  Creative (temp=1.2): {tagline_creative}")
print(f"  Factual  (temp=0.0): {tagline_factual}")

# ─────────────────────────────────────────────────────────────
# SECTION 5: Async LCEL — concurrent execution
# Use ainvoke/astream/abatch for async contexts (FastAPI, etc.)
# ─────────────────────────────────────────────────────────────
print("\n📌 5. Async LCEL")
print("─" * 40)

simple_chain = (
    ChatPromptTemplate.from_template("One fun fact about {topic}.")
    | llm | parser
)

async def async_demo():
    # ainvoke — single async call
    result = await simple_chain.ainvoke({"topic": "Python"})
    print(f"  ainvoke: {result[:80]}...")

    # abatch — multiple concurrent calls
    topics = [{"topic": t} for t in ["LangChain", "RAG", "LangGraph", "FAISS"]]
    results = await simple_chain.abatch(topics)
    print(f"  abatch ({len(results)} results):")
    for t, r in zip(topics, results):
        print(f"    [{t['topic']:<12}] {r[:60]}...")

    # astream — async token streaming
    print("  astream: ", end="", flush=True)
    async for chunk in simple_chain.astream({"topic": "neural networks"}):
        print(chunk, end="", flush=True)
    print()

asyncio.run(async_demo())

# ─────────────────────────────────────────────────────────────
# SECTION 6: Chain Inspection
# Every LCEL chain is inspectable — see its input/output schema.
# ─────────────────────────────────────────────────────────────
print("\n📌 6. Inspecting a Chain")
print("─" * 40)

print(f"  Input variables  : {simple_chain.input_schema.model_fields}")
print(f"  Output type      : {simple_chain.output_schema.__name__}")
print(f"  Steps in chain   : {len(simple_chain.steps)}")
for i, step in enumerate(simple_chain.steps):
    print(f"    [{i}] {type(step).__name__}")

# ─────────────────────────────────────────────────────────────
# ⚠️  COMMON MISTAKE 1: Forgetting fill="none" in lambdas
# RunnableLambda expects a function with exactly ONE argument.
# BAD:  RunnableLambda(lambda x, y: x + y)  ← 2 args!
# GOOD: RunnableLambda(lambda data: data["x"] + data["y"])

# ⚠️  COMMON MISTAKE 2: Using regular Python in async context
# If you're in an async function (FastAPI route), use ainvoke, not invoke.
# BAD:  result = chain.invoke(input)       ← blocks the event loop!
# GOOD: result = await chain.ainvoke(input)

# ⚠️  COMMON MISTAKE 3: Forgetting that parallel runs concurrently
# RunnableParallel sends ALL branches simultaneously.
# Don't put branches that depend on each other's output in parallel.
# ─────────────────────────────────────────────────────────────
print("\n⚠️  LCEL tips:")
print("   • RunnableLambda expects exactly ONE argument")
print("   • Use ainvoke/abatch in async contexts (FastAPI)")
print("   • RunnableParallel runs ALL branches concurrently")
print("   • Use configurable_fields() to make chains flexible")
