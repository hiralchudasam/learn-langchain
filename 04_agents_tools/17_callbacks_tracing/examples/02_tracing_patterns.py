"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 17 — Callbacks & Tracing                             ║
║  File:   02_tracing_patterns.py                              ║
║  Level:  Advanced                                            ║
║  Goal:   Async callbacks, structured event logging,          ║
║          latency monitoring, and production tracing          ║
╚══════════════════════════════════════════════════════════════╝
"""

from dotenv import load_dotenv
load_dotenv()

import time
import json
from datetime import datetime
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler, AsyncCallbackHandler
from langchain_core.outputs import LLMResult

llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

print("=" * 60)
print("17 — Advanced Tracing Patterns")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# PATTERN 1: Structured JSON logging
# Emit machine-readable logs for log aggregation (Datadog, ELK, etc.)
# ─────────────────────────────────────────────────────────────
print("\n📌 1. Structured JSON Logger")
print("─" * 40)

class StructuredLogger(BaseCallbackHandler):
    """Emits structured JSON log events for every LLM call."""

    def __init__(self, service_name: str = "langchain-app"):
        self.service_name = service_name
        self.logs = []  # store in memory for this demo

    def _emit(self, event: str, data: dict):
        log = {
            "timestamp":    datetime.utcnow().isoformat(),
            "service":      self.service_name,
            "event":        event,
            **data,
        }
        self.logs.append(log)
        print(f"  LOG: {json.dumps(log)}")

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._emit("llm_start", {
            "model": serialized.get("kwargs", {}).get("model_name", "?"),
            "prompt_length": sum(len(p) for p in prompts),
        })

    def on_llm_end(self, response: LLMResult, **kwargs):
        usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        self._emit("llm_end", {
            "input_tokens":  usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "finish_reason": response.generations[0][0].generation_info.get("finish_reason", "?") if response.generations else "?",
        })

    def on_llm_error(self, error: Exception, **kwargs):
        self._emit("llm_error", {"error": str(error)})

logger = StructuredLogger(service_name="langchain-tutorial")
chain  = (
    ChatPromptTemplate.from_template("What is {topic} in one sentence?")
    | llm | parser
)
result = chain.invoke({"topic": "LangChain"}, config={"callbacks": [logger]})
print(f"\n  Answer: {result}")
print(f"  Total log events emitted: {len(logger.logs)}")

# ─────────────────────────────────────────────────────────────
# PATTERN 2: Latency monitoring
# Measure time for each step, alert on slow calls.
# ─────────────────────────────────────────────────────────────
print("\n📌 2. Latency Monitor")
print("─" * 40)

class LatencyMonitor(BaseCallbackHandler):
    """Tracks and alerts on slow LLM calls."""

    SLOW_THRESHOLD_MS = 3000  # alert if > 3 seconds

    def __init__(self):
        self._timers = {}
        self.latencies = []

    def on_llm_start(self, serialized, prompts, run_id=None, **kwargs):
        self._timers[str(run_id)] = time.time()

    def on_llm_end(self, response, run_id=None, **kwargs):
        start = self._timers.pop(str(run_id), None)
        if start:
            ms = (time.time() - start) * 1000
            self.latencies.append(ms)
            status = "⚠️  SLOW" if ms > self.SLOW_THRESHOLD_MS else "✅ OK"
            print(f"  {status} | {ms:.0f}ms")

    def summary(self):
        if not self.latencies:
            return
        print(f"\n  Latency summary ({len(self.latencies)} calls):")
        print(f"    Min   : {min(self.latencies):.0f}ms")
        print(f"    Max   : {max(self.latencies):.0f}ms")
        print(f"    Avg   : {sum(self.latencies)/len(self.latencies):.0f}ms")
        slow = sum(1 for l in self.latencies if l > self.SLOW_THRESHOLD_MS)
        print(f"    Slow  : {slow}/{len(self.latencies)} calls")

monitor = LatencyMonitor()
topics  = ["Python", "RAG", "LangGraph", "embeddings"]
chain.batch(
    [{"topic": t} for t in topics],
    config={"callbacks": [monitor]},
)
monitor.summary()

# ─────────────────────────────────────────────────────────────
# PATTERN 3: Async callback for non-blocking I/O
# Use when your callback does I/O (write to DB, send to API).
# ─────────────────────────────────────────────────────────────
print("\n📌 3. Async Callback (non-blocking)")
print("─" * 40)

import asyncio

class AsyncDBLogger(AsyncCallbackHandler):
    """Logs to a 'database' asynchronously — doesn't block the chain."""

    async def on_llm_start(self, serialized, prompts, **kwargs):
        await asyncio.sleep(0.001)  # simulate async DB write
        print(f"  [AsyncDB] LLM started (async, non-blocking)")

    async def on_llm_end(self, response, **kwargs):
        await asyncio.sleep(0.001)  # simulate async DB write
        print(f"  [AsyncDB] LLM finished (async, non-blocking)")

async def run_with_async_callback():
    async_logger = AsyncDBLogger()
    result = await chain.ainvoke(
        {"topic": "async programming"},
        config={"callbacks": [async_logger]},
    )
    print(f"  Result: {result[:80]}")

asyncio.run(run_with_async_callback())

# ─────────────────────────────────────────────────────────────
# PATTERN 4: Chaining multiple callbacks together
# ─────────────────────────────────────────────────────────────
print("\n📌 4. Multiple Callbacks Simultaneously")
print("─" * 40)

class CounterCallback(BaseCallbackHandler):
    def __init__(self): self.count = 0
    def on_llm_end(self, *a, **kw): self.count += 1

counter1 = CounterCallback()
counter2 = LatencyMonitor()

# Both callbacks fire for every LLM call
chain.batch(
    [{"topic": "Python"}, {"topic": "LangChain"}],
    config={"callbacks": [counter1, counter2]},
)
print(f"  Counter1 (calls): {counter1.count}")
print(f"  Counter2 (latencies): {counter2.latencies}")
print(f"  ← Multiple callbacks all fire for the same run")

print("\n⚠️  Advanced tracing tips:")
print("   • Use AsyncCallbackHandler for I/O-bound callbacks (DB, HTTP)")
print("   • Emit structured JSON logs for log aggregation systems")
print("   • Monitor p95 latency — averages hide outliers")
print("   • Pass multiple callbacks as a list — they all fire")
