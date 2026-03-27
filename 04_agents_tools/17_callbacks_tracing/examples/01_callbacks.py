"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 17 — Callbacks & Tracing                             ║
║  File:   01_callbacks.py                                     ║
║  Level:  Basic → Advanced                                    ║
║  Goal:   Hook into the LangChain lifecycle, build custom     ║
║          loggers, track tokens, and debug chains             ║
╚══════════════════════════════════════════════════════════════╝
"""

from dotenv import load_dotenv
load_dotenv()

import time
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks import get_openai_callback

llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

print("=" * 60)
print("17 — Callbacks & Tracing")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# SECTION 1: What are callbacks?
# Callbacks let you hook into the lifecycle of a LangChain run.
# Every chain/model/tool fires events you can listen to:
#   on_chain_start → on_llm_start → on_llm_end → on_chain_end
# Great for: logging, cost tracking, debugging, monitoring.
# ─────────────────────────────────────────────────────────────
print("\n📌 1. Simple Logging Callback")
print("─" * 40)

class SimpleLogger(BaseCallbackHandler):
    """Logs every LLM call with timing."""

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        self._start = time.time()
        model = serialized.get("kwargs", {}).get("model_name", "unknown")
        print(f"  🚀 LLM started   | model={model} | prompts={len(prompts)}")

    def on_llm_end(self, response: LLMResult, **kwargs):
        elapsed = time.time() - self._start
        tokens  = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        print(f"  ✅ LLM finished  | time={elapsed:.2f}s | tokens={tokens.get('total_tokens', '?')}")

    def on_llm_error(self, error: Exception, **kwargs):
        print(f"  ❌ LLM error     | {error}")

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs):
        name = serialized.get("id", ["?"])[-1]
        print(f"  ▶️  Chain start   | {name}")

    def on_chain_end(self, outputs: dict, **kwargs):
        print(f"  ⏹️  Chain end")

# Pass callbacks at invocation time
logger = SimpleLogger()
chain  = (
    ChatPromptTemplate.from_template("Summarize {topic} in one sentence.")
    | llm | parser
)

print("  Invoking chain with SimpleLogger:")
result = chain.invoke({"topic": "LangChain"}, config={"callbacks": [logger]})
print(f"  Result: {result}")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Token & Cost Tracking Callback
# Track exact token usage and estimated cost per run.
# ─────────────────────────────────────────────────────────────
print("\n📌 2. Token & Cost Tracking")
print("─" * 40)

class CostTracker(BaseCallbackHandler):
    """Tracks token usage and estimates cost."""

    # OpenAI pricing (approximate, per 1M tokens)
    PRICES = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o":      {"input": 5.00, "output": 15.00},
    }

    def __init__(self):
        self.calls         = 0
        self.total_input   = 0
        self.total_output  = 0

    def on_llm_end(self, response: LLMResult, **kwargs):
        usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        self.calls        += 1
        self.total_input  += usage.get("prompt_tokens", 0)
        self.total_output += usage.get("completion_tokens", 0)

    @property
    def total_tokens(self):
        return self.total_input + self.total_output

    def cost_estimate(self, model: str = "gpt-4o-mini") -> float:
        prices = self.PRICES.get(model, self.PRICES["gpt-4o-mini"])
        return (self.total_input  / 1_000_000 * prices["input"] +
                self.total_output / 1_000_000 * prices["output"])

    def report(self):
        print(f"  Calls          : {self.calls}")
        print(f"  Input tokens   : {self.total_input:,}")
        print(f"  Output tokens  : {self.total_output:,}")
        print(f"  Total tokens   : {self.total_tokens:,}")
        print(f"  Est. cost      : ${self.cost_estimate():.6f} USD")

tracker = CostTracker()

topics = ["RAG", "LangGraph", "embeddings", "vector stores"]
chain.batch(
    [{"topic": t} for t in topics],
    config={"callbacks": [tracker]},
)

print(f"  Ran {len(topics)} queries:")
tracker.report()

# ─────────────────────────────────────────────────────────────
# SECTION 3: get_openai_callback — built-in cost tracker
# Simpler than a custom callback when you just want cost info.
# ─────────────────────────────────────────────────────────────
print("\n📌 3. get_openai_callback — Built-in Cost Tracking")
print("─" * 40)

with get_openai_callback() as cb:
    r1 = chain.invoke({"topic": "neural networks"})
    r2 = chain.invoke({"topic": "transformers"})

print(f"  Two calls made:")
print(f"  Prompt tokens    : {cb.prompt_tokens}")
print(f"  Completion tokens: {cb.completion_tokens}")
print(f"  Total tokens     : {cb.total_tokens}")
print(f"  Total cost       : ${cb.total_cost:.6f} USD")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Verbose Debugging Callback
# Prints every input and output at every step.
# Turn this on when debugging a misbehaving chain.
# ─────────────────────────────────────────────────────────────
print("\n📌 4. Debug Callback — Inspect All Inputs/Outputs")
print("─" * 40)

class DebugCallback(BaseCallbackHandler):
    """Prints full inputs and outputs at every step."""

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs):
        name = serialized.get("id", ["?"])[-1]
        print(f"  [START {name}] inputs: {str(inputs)[:100]}")

    def on_chain_end(self, outputs: dict, **kwargs):
        print(f"  [END]         outputs: {str(outputs)[:100]}")

    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"  [LLM]  prompt: {prompts[0][:80]}...")

    def on_llm_end(self, response, **kwargs):
        text = response.generations[0][0].text if response.generations else "?"
        print(f"  [LLM]  output: {text[:80]}...")

debug = DebugCallback()
chain.invoke({"topic": "LCEL"}, config={"callbacks": [debug]})

# ─────────────────────────────────────────────────────────────
# SECTION 5: Attaching callbacks globally (not per-invoke)
# Callbacks can be attached at the component level,
# so EVERY call automatically uses them.
# ─────────────────────────────────────────────────────────────
print("\n📌 5. Global Callbacks on a Component")
print("─" * 40)

always_log = SimpleLogger()

# Attach at LLM level — every call uses this callback
llm_with_logger = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    callbacks=[always_log],      # ← always active, no need to pass at invoke
)

chain_global = (
    ChatPromptTemplate.from_template("Define {concept} in 5 words.")
    | llm_with_logger
    | parser
)

# No config={"callbacks": ...} needed — it's baked in
result = chain_global.invoke({"concept": "embeddings"})
print(f"  Result: {result}")

# ─────────────────────────────────────────────────────────────
# ⚠️  COMMON MISTAKE 1: Forgetting to pass callbacks at invoke
# If you attach callbacks to the component, they always fire.
# If you pass at .invoke(), they only fire for that call.
# Use component-level for always-on monitoring,
# invoke-level for one-off debugging.

# ⚠️  COMMON MISTAKE 2: Callbacks are SYNCHRONOUS by default
# Heavy work in a callback (e.g. saving to DB) will slow down the chain.
# Use AsyncCallbackHandler for non-blocking callbacks.

# ⚠️  COMMON MISTAKE 3: Not logging errors
# Always implement on_llm_error / on_chain_error in production.
# Silent failures are hard to debug.
# ─────────────────────────────────────────────────────────────
print("\n⚠️  Callback tips:")
print("   • Use get_openai_callback() for quick cost checks")
print("   • Use component-level callbacks for always-on monitoring")
print("   • Implement on_llm_error — don't let errors go silent")
print("   • Use AsyncCallbackHandler for non-blocking logging")
