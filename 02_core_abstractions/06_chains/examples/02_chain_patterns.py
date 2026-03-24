"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 06 — Chains                                           ║
║  File:   02_chain_patterns.py                                ║
║  Level:  Advanced                                            ║
║  Goal:   Real-world chain patterns — multi-step pipelines,   ║
║          conditional chains, error handling in chains        ║
╚══════════════════════════════════════════════════════════════╝
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from pydantic import BaseModel, Field
from typing import List

llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
llm0   = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

print("=" * 60)
print("06 — Real-World Chain Patterns")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# PATTERN 1: Multi-Step Content Pipeline
# Each step enriches the output from the previous step.
# topic → outline → draft → improve → final
# ─────────────────────────────────────────────────────────────
print("\n📌 1. Multi-Step Content Pipeline")
print("─" * 40)

outline_chain = (
    ChatPromptTemplate.from_template(
        "Create a 3-point outline for a blog post about '{topic}'. "
        "Just list 3 headings, no details."
    )
    | llm | parser
)

draft_chain = (
    ChatPromptTemplate.from_template(
        "Write a short blog intro (2 sentences) using this outline:\n{outline}\n\nTopic: {topic}"
    )
    | llm | parser
)

improve_chain = (
    ChatPromptTemplate.from_template(
        "Rewrite this more engagingly and add one hook question:\n\n{draft}"
    )
    | llm | parser
)

def run_pipeline(topic: str) -> dict:
    outline = outline_chain.invoke({"topic": topic})
    draft   = draft_chain.invoke({"outline": outline, "topic": topic})
    final   = improve_chain.invoke({"draft": draft})
    return {"topic": topic, "outline": outline, "draft": draft, "final": final}

result = run_pipeline("LangChain for beginners")
print(f"  Topic  : {result['topic']}")
print(f"  Outline:\n{result['outline']}")
print(f"\n  Final  :\n{result['final']}")

# ─────────────────────────────────────────────────────────────
# PATTERN 2: Validation Chain
# Step 1: Generate something
# Step 2: Validate it
# Step 3: Fix it if needed
# ─────────────────────────────────────────────────────────────
print("\n📌 2. Generate → Validate → Fix Chain")
print("─" * 40)

class CodeReview(BaseModel):
    is_valid:  bool        = Field(description="Is the code correct Python?")
    issues:    List[str]   = Field(description="List of issues found, empty if valid")
    fixed_code: str        = Field(description="The corrected code, same as input if valid")

code_gen_chain = (
    ChatPromptTemplate.from_template("Write a Python function to {task}. Return only the code.")
    | llm | parser
)

review_parser = PydanticOutputParser(pydantic_object=CodeReview)
review_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a Python code reviewer. {format_instructions}"),
        ("human", "Review this code:\n\n```python\n{code}\n```"),
    ]).partial(format_instructions=review_parser.get_format_instructions())
    | llm0 | review_parser
)

def generate_and_validate(task: str) -> dict:
    code   = code_gen_chain.invoke({"task": task})
    review = review_chain.invoke({"code": code})
    return {"original": code, "review": review}

result = generate_and_validate("calculate fibonacci numbers recursively")
review = result["review"]
print(f"  Valid  : {review.is_valid}")
print(f"  Issues : {review.issues or 'None'}")
print(f"  Code   :\n{review.fixed_code[:200]}")

# ─────────────────────────────────────────────────────────────
# PATTERN 3: Summarize → Translate chain
# Practical pipeline: process content, then localize it.
# ─────────────────────────────────────────────────────────────
print("\n📌 3. Summarize → Translate Pipeline")
print("─" * 40)

SAMPLE_TEXT = """
LangChain is a framework for building applications powered by language models.
It provides a standard interface for chains, agents, and retrieval strategies.
LangChain supports multiple LLM providers including OpenAI, Anthropic, and local models.
The framework uses LCEL (LangChain Expression Language) for composing chains with the pipe operator.
"""

summarize = (
    ChatPromptTemplate.from_template("Summarize in 2 sentences:\n\n{text}")
    | llm | parser
)
translate = (
    ChatPromptTemplate.from_template("Translate to {language}:\n\n{summary}")
    | llm | parser
)

# Chain them: text → summary → translated summary
def summarize_and_translate(text: str, language: str) -> dict:
    summary     = summarize.invoke({"text": text})
    translated  = translate.invoke({"summary": summary, "language": language})
    return {"summary": summary, "translated": translated}

for lang in ["Hindi", "Spanish", "French"]:
    r = summarize_and_translate(SAMPLE_TEXT, lang)
    print(f"\n  [{lang}] {r['translated'][:100]}...")

# ─────────────────────────────────────────────────────────────
# PATTERN 4: Retry / Error-handling in a chain
# Wrap any step with retry logic for production resilience.
# ─────────────────────────────────────────────────────────────
print("\n📌 4. Error Handling in Chains")
print("─" * 40)

attempt_n = 0

def unreliable_step(text: str) -> str:
    """Simulates a step that fails on the first attempt."""
    global attempt_n
    attempt_n += 1
    if attempt_n == 1:
        raise ValueError("Simulated transient failure on attempt 1")
    attempt_n = 0
    return text.upper()

resilient_chain = (
    ChatPromptTemplate.from_template("Summarize {topic} in one sentence.")
    | llm | parser
    | RunnableLambda(unreliable_step).with_retry(
        retry_if_exception_type=(ValueError,),
        stop_after_attempt=3,
    )
)

result = resilient_chain.invoke({"topic": "LangChain"})
print(f"  Result (after retry): {result[:80]}...")
print(f"  The chain auto-retried the failing step!")

# ─────────────────────────────────────────────────────────────
# PATTERN 5: Parallel Research → Synthesis chain
# Gather information from multiple angles, then synthesize.
# ─────────────────────────────────────────────────────────────
print("\n📌 5. Research → Synthesis")
print("─" * 40)

research = RunnableParallel(
    definition = (ChatPromptTemplate.from_template("Define {topic} in one sentence.") | llm | parser),
    history    = (ChatPromptTemplate.from_template("When was {topic} created and by whom?") | llm | parser),
    use_cases  = (ChatPromptTemplate.from_template("Name 3 main use cases for {topic}.") | llm | parser),
)

def synthesize(data: dict) -> str:
    synth_prompt = ChatPromptTemplate.from_template(
        "Write a 3-sentence expert overview of {topic} using these notes:\n\n"
        "Definition: {definition}\n\nHistory: {history}\n\nUse Cases: {use_cases}"
    )
    return (synth_prompt | llm | parser).invoke(data)

research_chain = (
    research
    | RunnableLambda(lambda d: {**d, "topic": "LangChain"})
    | RunnableLambda(synthesize)
)

overview = research_chain.invoke({"topic": "LangChain"})
print(f"  {overview}")

# ─────────────────────────────────────────────────────────────
# ⚠️  COMMON MISTAKE 1: Assuming chains run sequentially
# RunnableParallel branches run CONCURRENTLY.
# Don't assume any ordering within a parallel block.

# ⚠️  COMMON MISTAKE 2: Not passing all required keys through
# If step B needs "topic" but step A only outputs "summary",
# use RunnablePassthrough or RunnableParallel to carry forward.

# ⚠️  COMMON MISTAKE 3: Catching exceptions OUTSIDE the chain
# Use .with_retry() or .with_fallbacks() INSIDE the chain.
# Wrapping in try/except outside the chain loses context.
# ─────────────────────────────────────────────────────────────
print("\n⚠️  Chain tips:")
print("   • Parallel branches run CONCURRENTLY — no ordering guarantee")
print("   • Use RunnablePassthrough to carry inputs through steps")
print("   • Use .with_retry() inside the chain for resilience")
