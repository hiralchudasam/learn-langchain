"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 02 — Language Models                                  ║
║  File:   02_model_deep_dive.py                               ║
║  Level:  Intermediate → Advanced                             ║
║  Goal:   Explore model parameters, multiple providers,       ║
║          async calls, and structured output                  ║
╚══════════════════════════════════════════════════════════════╝
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

print("=" * 60)
print("02 — Language Models Deep Dive")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# SECTION 1: Temperature — controls randomness
# 0.0 = always the same answer (best for facts, JSON, code)
# 0.7 = balanced (best for general use)
# 1.5+ = very creative / unpredictable (poems, brainstorming)
# ─────────────────────────────────────────────────────────────
print("\n📌 1. Temperature Effect")
print("─" * 40)

question = [HumanMessage(content="Give me one creative word for 'happy'.")]

for temp in [0.0, 0.7, 1.5]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=temp)
    response = llm.invoke(question)
    print(f"  temperature={temp} → {response.content.strip()}")

# ─────────────────────────────────────────────────────────────
# SECTION 2: max_tokens — limit output length
# Useful to control costs and prevent runaway responses.
# ─────────────────────────────────────────────────────────────
print("\n📌 2. max_tokens — Controlling Output Length")
print("─" * 40)

short_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=20)
long_llm  = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=200)

q = [HumanMessage(content="Explain what LangChain is.")]
print(f"  max_tokens=20  → {short_llm.invoke(q).content}")
print(f"  max_tokens=200 → {long_llm.invoke(q).content[:100]}...")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Response Metadata
# Every response carries token counts, model name, finish reason.
# Use this for cost tracking and debugging.
# ─────────────────────────────────────────────────────────────
print("\n📌 3. Response Metadata — Token Usage")
print("─" * 40)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
response = llm.invoke([HumanMessage(content="What is RAG in LLMs?")])

print(f"  Content         : {response.content[:80]}...")
print(f"  Model           : {response.response_metadata.get('model_name')}")
print(f"  Finish reason   : {response.response_metadata.get('finish_reason')}")
print(f"  Input tokens    : {response.usage_metadata.get('input_tokens')}")
print(f"  Output tokens   : {response.usage_metadata.get('output_tokens')}")
print(f"  Total tokens    : {response.usage_metadata.get('total_tokens')}")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Streaming — receive tokens as they generate
# Great for building responsive UIs and CLI chatbots.
# ─────────────────────────────────────────────────────────────
print("\n📌 4. Streaming Tokens")
print("─" * 40)

stream_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

print("  Response: ", end="", flush=True)
full_text = ""
for chunk in stream_llm.stream([HumanMessage(content="Write a 2-sentence poem about AI.")]):
    print(chunk.content, end="", flush=True)
    full_text += chunk.content
print(f"\n  Total chars: {len(full_text)}")

# ─────────────────────────────────────────────────────────────
# SECTION 5: Async calls — run multiple LLM calls concurrently
# Much faster than sequential calls when you have many prompts.
# ─────────────────────────────────────────────────────────────
print("\n📌 5. Async Calls — Concurrent Execution")
print("─" * 40)

async def run_async_demo():
    import time
    questions = [
        [HumanMessage(content="Capital of France?")],
        [HumanMessage(content="Capital of Japan?")],
        [HumanMessage(content="Capital of Brazil?")],
        [HumanMessage(content="Capital of India?")],
    ]

    # Sequential (slow)
    start = time.time()
    for q in questions:
        llm.invoke(q)
    sequential_time = time.time() - start

    # Concurrent with batch (fast — runs in parallel)
    start = time.time()
    responses = await llm.abatch(questions)
    concurrent_time = time.time() - start

    print(f"  Sequential time : {sequential_time:.2f}s")
    print(f"  Concurrent time : {concurrent_time:.2f}s")
    print(f"  Speedup         : {sequential_time/concurrent_time:.1f}x faster")

    for q, r in zip(questions, responses):
        print(f"  {q[0].content:<25} → {r.content.strip()}")

asyncio.run(run_async_demo())

# ─────────────────────────────────────────────────────────────
# SECTION 6: with_structured_output — get typed Pydantic objects
# The modern way to get structured data from an LLM.
# No manual JSON parsing needed.
# ─────────────────────────────────────────────────────────────
print("\n📌 6. Structured Output — Typed Pydantic Responses")
print("─" * 40)

class MovieReview(BaseModel):
    title:    str   = Field(description="Movie title")
    year:     int   = Field(description="Release year")
    rating:   float = Field(description="Rating out of 10")
    summary:  str   = Field(description="One-sentence summary")
    worth_watching: bool = Field(description="Would you recommend it?")

structured_llm = llm.with_structured_output(MovieReview)
review = structured_llm.invoke("Tell me about the movie Inception")

print(f"  Title          : {review.title}")
print(f"  Year           : {review.year}")
print(f"  Rating         : {review.rating}/10")
print(f"  Summary        : {review.summary}")
print(f"  Worth watching : {review.worth_watching}")
print(f"  Python type    : {type(review).__name__}  ← real Pydantic object!")

# ─────────────────────────────────────────────────────────────
# ⚠️  COMMON MISTAKE 1: Using temperature=0 for creative tasks
# temperature=0 produces the same answer every time.
# Use temperature=0 only for facts, code, and structured output.

# ⚠️  COMMON MISTAKE 2: Not tracking token usage
# Every token costs money. Always log usage_metadata in production.

# ⚠️  COMMON MISTAKE 3: Calling .invoke() in a loop (slow!)
# Use .batch() or .abatch() for multiple inputs — they run in parallel.
# BAD:  for q in questions: llm.invoke(q)
# GOOD: llm.batch(questions)
# ─────────────────────────────────────────────────────────────
print("\n⚠️  Key reminders:")
print("   • temperature=0   for facts/code/JSON")
print("   • temperature=0.7 for general use")
print("   • Use .batch()    instead of looping .invoke()")
print("   • Always log      usage_metadata in production")
