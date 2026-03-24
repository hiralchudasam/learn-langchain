"""
05 — LangChain Expression Language (LCEL)
Example 01: Pipe operator, Parallel, Passthrough, Lambda, Branch
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
    RunnableBranch,
)

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

# ─── 1. Basic Pipe Chain ──────────────────────────────────────────────────────

basic_chain = (
    ChatPromptTemplate.from_template("Tell me a fact about {topic}.")
    | llm
    | parser
)

print("Basic chain:", basic_chain.invoke({"topic": "black holes"}))

# ─── 2. RunnablePassthrough ───────────────────────────────────────────────────
# Adds the original question alongside the answer

qa_chain = RunnableParallel(
    question=RunnablePassthrough(),
    answer=(
        ChatPromptTemplate.from_template("Answer briefly: {question}")
        | llm
        | parser
    ),
)

result = qa_chain.invoke({"question": "What is a neural network?"})
print("\nPassthrough:")
print("  Q:", result["question"])
print("  A:", result["answer"])

# ─── 3. RunnableParallel (same input, multiple outputs) ───────────────────────

parallel_chain = RunnableParallel(
    pros=(
        ChatPromptTemplate.from_template("List 3 pros of {language} in one line each.")
        | llm | parser
    ),
    cons=(
        ChatPromptTemplate.from_template("List 3 cons of {language} in one line each.")
        | llm | parser
    ),
    use_cases=(
        ChatPromptTemplate.from_template("List 3 use cases for {language} in one line each.")
        | llm | parser
    ),
)

result = parallel_chain.invoke({"language": "Python"})
print("\nParallel chain:")
print("  Pros:", result["pros"][:80], "...")
print("  Cons:", result["cons"][:80], "...")
print("  Uses:", result["use_cases"][:80], "...")

# ─── 4. RunnableLambda ────────────────────────────────────────────────────────

def format_as_bullets(text: str) -> str:
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return "\n".join(f"• {line}" for line in lines)

bullet_chain = (
    ChatPromptTemplate.from_template("List 4 tips for learning {topic}. One per line.")
    | llm
    | parser
    | RunnableLambda(format_as_bullets)
)

print("\nRunnableLambda (bullet formatter):")
print(bullet_chain.invoke({"topic": "machine learning"}))

# ─── 5. RunnableBranch ────────────────────────────────────────────────────────

python_chain = (
    ChatPromptTemplate.from_template("Explain this Python concept: {topic}")
    | llm | parser
)
sql_chain = (
    ChatPromptTemplate.from_template("Explain this SQL concept: {topic}")
    | llm | parser
)
general_chain = (
    ChatPromptTemplate.from_template("Explain this programming concept: {topic}")
    | llm | parser
)

branch = RunnableBranch(
    (lambda x: "python" in x["topic"].lower(), python_chain),
    (lambda x: "sql"    in x["topic"].lower(), sql_chain),
    general_chain,
)

print("\nBranch — Python topic:", branch.invoke({"topic": "Python decorators"})[:80])
print("Branch — SQL topic:", branch.invoke({"topic": "SQL joins"})[:80])
print("Branch — General:", branch.invoke({"topic": "recursion"})[:80])

# ─── 6. Streaming ─────────────────────────────────────────────────────────────

stream_chain = (
    ChatPromptTemplate.from_template("Write a 3-sentence story about {character}.")
    | llm
    | parser
)

print("\nStreaming:")
for chunk in stream_chain.stream({"character": "a robot who learns to code"}):
    print(chunk, end="", flush=True)
print()

# ─── 7. Batch ─────────────────────────────────────────────────────────────────

topics = [{"topic": "Mars"}, {"topic": "Jupiter"}, {"topic": "Saturn"}]
results = basic_chain.batch(topics)
for topic, result in zip(topics, results):
    print(f"\nBatch — {topic['topic']}: {result[:60]}...")
