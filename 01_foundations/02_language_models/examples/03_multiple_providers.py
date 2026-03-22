"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 02 — Language Models                                  ║
║  File:   03_multiple_providers.py                            ║
║  Level:  Advanced                                            ║
║  Goal:   Use multiple LLM providers with the SAME chain.     ║
║          Demonstrates provider-agnostic LangChain design.    ║
╚══════════════════════════════════════════════════════════════╝
"""

from dotenv import load_dotenv
load_dotenv()

import os
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel

print("=" * 60)
print("02 — Multiple LLM Providers")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# KEY INSIGHT: LangChain's power — swap the LLM, keep the chain.
# Every provider implements the same BaseChatModel interface.
# Your prompt, parser, and logic never change.
# ─────────────────────────────────────────────────────────────

# Same prompt and parser for all providers
prompt = ChatPromptTemplate.from_template(
    "In exactly one sentence, explain: {concept}"
)
parser = StrOutputParser()

def run_with_model(model: BaseChatModel, name: str, concept: str) -> dict:
    """Run the same chain with any LLM provider."""
    chain = prompt | model | parser
    start = time.time()
    result = chain.invoke({"concept": concept})
    elapsed = time.time() - start
    return {"model": name, "answer": result, "time": elapsed}

# ─────────────────────────────────────────────────────────────
# PROVIDER 1: OpenAI (GPT models)
# ─────────────────────────────────────────────────────────────
print("\n📌 1. OpenAI Provider")
print("─" * 40)

models_openai = {
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", temperature=0),
    "gpt-4o":      ChatOpenAI(model="gpt-4o",      temperature=0),
}

for model_name, model in models_openai.items():
    result = run_with_model(model, model_name, "gradient descent")
    print(f"  [{result['model']:<15}] ({result['time']:.2f}s) {result['answer'][:80]}")

# ─────────────────────────────────────────────────────────────
# PROVIDER 2: Anthropic Claude (if API key is set)
# ─────────────────────────────────────────────────────────────
print("\n📌 2. Anthropic Claude Provider")
print("─" * 40)

if os.getenv("ANTHROPIC_API_KEY"):
    from langchain_anthropic import ChatAnthropic
    claude = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
    result = run_with_model(claude, "claude-3-haiku", "gradient descent")
    print(f"  [{result['model']:<20}] ({result['time']:.2f}s) {result['answer'][:80]}")
else:
    print("  ⏭️  Skipped — set ANTHROPIC_API_KEY in .env to test Claude")

# ─────────────────────────────────────────────────────────────
# PROVIDER 3: Ollama (local models, no API key needed)
# Install Ollama: https://ollama.ai  then: ollama pull llama3.2
# ─────────────────────────────────────────────────────────────
print("\n📌 3. Ollama Local Model (offline, free)")
print("─" * 40)

try:
    from langchain_ollama import ChatOllama
    ollama = ChatOllama(model="llama3.2", temperature=0)
    result = run_with_model(ollama, "llama3.2-local", "gradient descent")
    print(f"  [{result['model']:<20}] ({result['time']:.2f}s) {result['answer'][:80]}")
except Exception:
    print("  ⏭️  Skipped — install Ollama and run: ollama pull llama3.2")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Model Comparison — run all on same question
# ─────────────────────────────────────────────────────────────
print("\n📌 4. Side-by-Side Comparison")
print("─" * 40)

concept = "attention mechanism in transformers"
print(f"  Question: 'Explain {concept} in one sentence'\n")

available_models = {
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", temperature=0),
    "gpt-4o":      ChatOpenAI(model="gpt-4o",      temperature=0),
}

for name, model in available_models.items():
    r = run_with_model(model, name, concept)
    print(f"  {r['model']}")
    print(f"  ↳ {r['answer']}")
    print(f"  ↳ Time: {r['time']:.2f}s\n")

# ─────────────────────────────────────────────────────────────
# SECTION 5: Model Fallback pattern
# If the primary model fails (rate limit, outage),
# automatically try a cheaper/backup model.
# ─────────────────────────────────────────────────────────────
print("\n📌 5. Fallback — Primary → Backup Model")
print("─" * 40)

primary  = ChatOpenAI(model="gpt-4o",      temperature=0)
fallback = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# .with_fallbacks() tries primary first, falls back on any exception
robust_llm = primary.with_fallbacks([fallback])
robust_chain = prompt | robust_llm | parser

result = robust_chain.invoke({"concept": "embeddings"})
print(f"  Result (with fallback safety): {result}")
print(f"  If gpt-4o fails → gpt-4o-mini takes over automatically")

# ─────────────────────────────────────────────────────────────
# ⚠️  COMMON MISTAKE: Assuming all providers support all features
# Not all models support:
#   - tool/function calling  (check model docs)
#   - vision/images          (needs gpt-4o or claude-3+)
#   - structured output      (works best with OpenAI)
#   - streaming              (all support it, but speed varies)
# ─────────────────────────────────────────────────────────────
print("\n⚠️  Provider capability matrix:")
print("   Feature              GPT-4o  GPT-4o-mini  Claude-3  Llama3")
print("   Tool calling           ✅       ✅           ✅        ✅")
print("   Vision (images)        ✅       ✅           ✅        ❌")
print("   Structured output      ✅       ✅           ✅        ⚠️")
print("   Context window         128K     128K         200K      128K")
print("   Cost (approx)          $$$      $            $$        Free")
