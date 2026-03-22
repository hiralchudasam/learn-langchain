"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 03 — Prompt Templates                                 ║
║  File:   02_advanced_prompting.py                            ║
║  Level:  Advanced                                            ║
║  Goal:   Dynamic prompts, prompt composition, system         ║
║          prompt patterns, and prompt best practices          ║
╚══════════════════════════════════════════════════════════════╝
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

print("=" * 60)
print("03 — Advanced Prompt Templates")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# SECTION 1: System Prompt Patterns
# The system prompt sets the model's role, tone, and constraints.
# It's the most powerful way to control LLM behaviour.
# ─────────────────────────────────────────────────────────────
print("\n📌 1. System Prompt Patterns")
print("─" * 40)

# Pattern A: Role + Constraint
template_a = ChatPromptTemplate.from_messages([
    ("system", "You are a senior Python developer. "
               "Always answer with working code examples. "
               "Keep explanations under 3 sentences."),
    ("human", "{question}"),
])

# Pattern B: Persona + Output format
template_b = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly tutor for 10-year-olds. "
               "Use simple words, analogies, and emojis. "
               "Always end with a fun fact."),
    ("human", "{question}"),
])

question = "What is a variable?"
chain_a = template_a | llm | parser
chain_b = template_b | llm | parser

print("  [Developer persona]:")
print(f"  {chain_a.invoke({'question': question})[:150]}...\n")
print("  [Kids tutor persona]:")
print(f"  {chain_b.invoke({'question': question})[:150]}...")

# ─────────────────────────────────────────────────────────────
# SECTION 2: FewShotChatMessagePromptTemplate
# Show the model examples of exactly the style you want.
# Dramatically improves output consistency.
# ─────────────────────────────────────────────────────────────
print("\n📌 2. Few-Shot Chat Prompting")
print("─" * 40)

# Define examples of the input → output style you want
examples = [
    {"input": "slow",   "output": "fast"},
    {"input": "dark",   "output": "light"},
    {"input": "loud",   "output": "quiet"},
    {"input": "rough",  "output": "smooth"},
]

# Each example becomes a HumanMessage → AIMessage pair
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai",    "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# Wrap in a full chat template
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You give antonyms. Reply with ONE word only."),
    few_shot_prompt,          # ← inject the examples here
    ("human", "{word}"),
])

few_shot_chain = final_prompt | llm | parser

test_words = ["hot", "happy", "tall", "brave"]
print(f"  Few-shot antonyms:")
for word in test_words:
    result = few_shot_chain.invoke({"word": word})
    print(f"    {word:<10} → {result.strip()}")

# ─────────────────────────────────────────────────────────────
# SECTION 3: MessagesPlaceholder — inject conversation history
# This is how you add memory to any chain.
# ─────────────────────────────────────────────────────────────
print("\n📌 3. MessagesPlaceholder — Inject History")
print("─" * 40)

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Remember everything the user tells you."),
    MessagesPlaceholder("history"),  # ← conversation history injected here
    ("human", "{input}"),
])

chain = chat_template | llm | parser
history = []

def chat(user_input: str) -> str:
    response = chain.invoke({"input": user_input, "history": history})
    # Append to history after each turn
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response))
    return response

print("  Turn 1:", chat("My name is Rahul and I work at Google."))
print("  Turn 2:", chat("What's my name and where do I work?"))

# ─────────────────────────────────────────────────────────────
# SECTION 4: Partial Templates — pre-fill some variables
# Lock in certain values, keep others dynamic.
# Great for creating specialized sub-chains from a generic template.
# ─────────────────────────────────────────────────────────────
print("\n📌 4. Partial Templates")
print("─" * 40)

base = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Always respond in {language}."),
    ("human", "{question}"),
])

# Create specialized chains by partially filling the template
english_developer = (base.partial(role="Python developer", language="English")
                     | llm | parser)
hindi_teacher     = (base.partial(role="school teacher",   language="Hindi")
                     | llm | parser)

q = "What is a loop?"
print(f"  [English developer]: {english_developer.invoke({'question': q})[:100]}...")
print(f"  [Hindi teacher]    : {hindi_teacher.invoke({'question': q})[:100]}...")

# ─────────────────────────────────────────────────────────────
# SECTION 5: Dynamic Prompt Selection
# Choose which prompt template to use based on input content.
# ─────────────────────────────────────────────────────────────
print("\n📌 5. Dynamic Prompt Selection")
print("─" * 40)

code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a coding assistant. Provide working code with comments."),
    ("human", "{question}"),
])

theory_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a CS theory teacher. Explain concepts clearly without code."),
    ("human", "{question}"),
])

def smart_chain(question: str) -> str:
    # Pick prompt based on question content
    code_keywords = ["code", "implement", "write", "function", "class", "script"]
    if any(kw in question.lower() for kw in code_keywords):
        prompt_to_use = code_prompt
        mode = "code"
    else:
        prompt_to_use = theory_prompt
        mode = "theory"

    chain = prompt_to_use | llm | parser
    result = chain.invoke({"question": question})
    return f"[{mode} mode] {result[:120]}..."

print(smart_chain("What is recursion?"))
print(smart_chain("Write a recursive factorial function"))

# ─────────────────────────────────────────────────────────────
# ⚠️  COMMON MISTAKE 1: Putting too much in the system prompt
# Long system prompts reduce context window for actual content.
# Keep system prompts focused — max 200 words.

# ⚠️  COMMON MISTAKE 2: Not using few-shot for structured output
# If the model keeps returning the wrong format, add 2-3 examples.
# Few-shot examples are the most reliable format enforcement.

# ⚠️  COMMON MISTAKE 3: Forgetting MessagesPlaceholder for history
# If you don't inject history, the model won't remember past turns.
# ─────────────────────────────────────────────────────────────
print("\n⚠️  Prompt engineering tips:")
print("   • Keep system prompts under 200 words")
print("   • Use few-shot examples for consistent output format")
print("   • Use MessagesPlaceholder for conversation history")
print("   • Use partial() to create reusable specialized chains")
