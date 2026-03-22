"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 01 — Introduction to LangChain                       ║
║  File:   01_hello_langchain.py                               ║
║  Level:  Basic                                               ║
║  Goal:   Your very first LangChain program — understand      ║
║          the core building blocks: model, prompt, parser     ║
╚══════════════════════════════════════════════════════════════╝
"""

# ── Step 1: Load environment variables from .env ──────────────────────────────
# Always do this FIRST before importing LangChain components
from dotenv import load_dotenv
load_dotenv()  # reads OPENAI_API_KEY from .env file

# ── Step 2: Import the three core building blocks ─────────────────────────────
from langchain_openai import ChatOpenAI                    # The LLM
from langchain_core.prompts import ChatPromptTemplate      # The prompt
from langchain_core.output_parsers import StrOutputParser  # The parser

print("=" * 60)
print("01 — Hello LangChain!")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# BUILDING BLOCK 1: The Model
# ChatOpenAI wraps OpenAI's chat API.
# model     → which GPT model to use
# temperature → 0 = deterministic, 1+ = creative/random
# ─────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4o-mini",  # cheapest + fast, great for learning
    temperature=0.7,
)
print("\n✅ Model created:", llm.model_name)

# ─────────────────────────────────────────────────────────────
# BUILDING BLOCK 2: The Prompt Template
# A reusable template with {placeholders} that get filled at runtime.
# Much better than f-strings — validates inputs automatically.
# ─────────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in exactly 2 sentences, for a complete beginner."
)
print("✅ Prompt created. Variables:", prompt.input_variables)

# ─────────────────────────────────────────────────────────────
# BUILDING BLOCK 3: The Output Parser
# StrOutputParser extracts the plain string from the AIMessage response.
# Without it, you'd get back an AIMessage object, not a plain string.
# ─────────────────────────────────────────────────────────────
parser = StrOutputParser()
print("✅ Parser created: StrOutputParser")

# ─────────────────────────────────────────────────────────────
# CONNECTING THEM: The Chain (using LCEL pipe operator |)
# prompt | llm | parser
#   1. prompt formats the input into messages
#   2. llm sends messages to OpenAI and gets a response
#   3. parser extracts the text string from the response
# ─────────────────────────────────────────────────────────────
chain = prompt | llm | parser
print("\n✅ Chain assembled: prompt | llm | parser")

# ─────────────────────────────────────────────────────────────
# RUNNING THE CHAIN
# .invoke() runs the chain once with the given input dict
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("Running chain...")
print("─" * 60)

topics = ["LangChain", "RAG", "LangGraph", "vector databases"]

for topic in topics:
    result = chain.invoke({"topic": topic})
    print(f"\n📌 {topic}:")
    print(f"   {result}")

# ─────────────────────────────────────────────────────────────
# ⚠️  COMMON MISTAKE 1: Forgetting load_dotenv()
# If you skip load_dotenv(), the API key won't be found.
# Error: openai.AuthenticationError: No API key provided
# Fix: Always call load_dotenv() at the top of your script.
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# ⚠️  COMMON MISTAKE 2: Wrong input key name
# chain.invoke({"Topic": "LangChain"})  ← capital T → KeyError!
# chain.invoke({"topic": "LangChain"})  ← must match template placeholder
# ─────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("✅ Done! You've run your first LangChain chain.")
print("─" * 60)
