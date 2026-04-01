"""
21 — Advanced Patterns
Example 01: Caching, fallbacks, retry, guardrails, cost tracking
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache, SQLiteCache
from langchain_community.callbacks import get_openai_callback
import time

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

chain = (
    ChatPromptTemplate.from_template("What is {topic} in one sentence?")
    | llm
    | parser
)

# ─── 1. In-Memory Caching ─────────────────────────────────────────────────────

print("=" * 55)
print("1. In-Memory Cache")
print("=" * 55)

set_llm_cache(InMemoryCache())

start = time.time()
r1 = chain.invoke({"topic": "LangChain"})
t1 = time.time() - start
print(f"First call  ({t1:.2f}s): {r1}")

start = time.time()
r2 = chain.invoke({"topic": "LangChain"})   # exact same → cache hit
t2 = time.time() - start
print(f"Second call ({t2:.2f}s): {r2}")
print(f"Speedup: {t1/max(t2, 0.001):.0f}x faster (cache hit)")

# ─── 2. SQLite Cache (persistent) ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("2. SQLite Cache (persists across restarts)")
print("=" * 55)

set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))
r = chain.invoke({"topic": "RAG"})
print(f"Cached to disk: {r}")
print("→ Re-run this script and the second call will use disk cache")

# Disable cache for remaining examples
set_llm_cache(None)

# ─── 3. Fallbacks ─────────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("3. Fallbacks")
print("=" * 55)

primary_llm = ChatOpenAI(model="gpt-4o", temperature=0)
fallback_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

robust_llm = primary_llm.with_fallbacks([fallback_llm])

robust_chain = (
    ChatPromptTemplate.from_template("Explain {topic} briefly.")
    | robust_llm
    | parser
)

result = robust_chain.invoke({"topic": "neural networks"})
print(f"Robust chain result: {result[:100]}...")
print("→ If gpt-4o fails/times out, gpt-4o-mini takes over automatically")

# ─── 4. Retry Logic ───────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("4. Retry Logic")
print("=" * 55)

from langchain_core.runnables import RunnableRetry

attempt_count = 0

def flaky_function(x: dict) -> str:
    """Simulates a function that fails the first 2 times."""
    global attempt_count
    attempt_count += 1
    if attempt_count < 3:
        raise ValueError(f"Simulated failure (attempt {attempt_count})")
    attempt_count = 0
    return f"Success on attempt 3: {x}"

flaky_runnable = RunnableLambda(flaky_function).with_retry(
    retry_if_exception_type=(ValueError,),
    stop_after_attempt=5,
    wait_exponential_jitter=False,
)

result = flaky_runnable.invoke({"input": "test"})
print(f"Retry result: {result}")

# ─── 5. Guardrails ────────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("5. Guardrails (Input Validation)")
print("=" * 55)

BLOCKED_WORDS = ["hack", "exploit", "malware", "illegal", "crack"]

def input_guard(inputs: dict) -> dict:
    text = inputs.get("topic", "").lower()
    for word in BLOCKED_WORDS:
        if word in text:
            raise ValueError(f"Blocked input: contains disallowed term '{word}'")
    return inputs

def output_guard(output: str) -> str:
    if len(output) < 10:
        raise ValueError("Output too short — may be an error response")
    return output

safe_chain = (
    RunnableLambda(input_guard)
    | chain
    | RunnableLambda(output_guard)
)

# Safe input
result = safe_chain.invoke({"topic": "machine learning"})
print(f"Safe input OK: {result[:60]}...")

# Blocked input
try:
    safe_chain.invoke({"topic": "how to hack a system"})
except ValueError as e:
    print(f"Blocked: {e}")

# ─── 6. Cost Tracking ─────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("6. Cost Tracking")
print("=" * 55)

set_llm_cache(None)

topics = ["LangChain", "RAG", "LangGraph", "embeddings", "vector stores"]

with get_openai_callback() as cb:
    results = chain.batch([{"topic": t} for t in topics])

print(f"Ran {len(topics)} queries:")
print(f"  Prompt tokens:     {cb.prompt_tokens:,}")
print(f"  Completion tokens: {cb.completion_tokens:,}")
print(f"  Total tokens:      {cb.total_tokens:,}")
print(f"  Estimated cost:    ${cb.total_cost:.4f} USD")
print(f"  Cost per query:    ${cb.total_cost/len(topics):.4f} USD")

# ─── 7. Configurable Chains ───────────────────────────────────────────────────

print("\n" + "=" * 55)
print("7. Configurable Runnables (swap model at runtime)")
print("=" * 55)

from langchain_core.runnables import ConfigurableField

configurable_llm = ChatOpenAI(model="gpt-4o-mini").configurable_fields(
    model_name=ConfigurableField(id="model", name="Model Name"),
    temperature=ConfigurableField(id="temperature", name="Temperature"),
)

configurable_chain = (
    ChatPromptTemplate.from_template("Tell me a fact about {topic}.")
    | configurable_llm
    | parser
)

result_mini = configurable_chain.invoke(
    {"topic": "Python"},
    config={"configurable": {"model": "gpt-4o-mini", "temperature": 0.0}},
)
print(f"gpt-4o-mini (temp=0.0): {result_mini}")

result_creative = configurable_chain.invoke(
    {"topic": "Python"},
    config={"configurable": {"model": "gpt-4o-mini", "temperature": 1.2}},
)
print(f"gpt-4o-mini (temp=1.2): {result_creative}")

# Cleanup
import os
if os.path.exists(".langchain_cache.db"):
    os.remove(".langchain_cache.db")
    print("\nCleaned up cache file.")
