"""
06 — Chains
Example 01: Sequential chain, Map-Reduce, Router with LCEL
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableBranch

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

# ─── 1. Sequential Chain (output feeds into next) ─────────────────────────────

print("=" * 55)
print("1. Sequential Chain: Topic → Outline → Article")
print("=" * 55)

outline_prompt = ChatPromptTemplate.from_template(
    "Create a 3-point outline for a blog post about: {topic}\nJust list the 3 headings, no details."
)
article_prompt = ChatPromptTemplate.from_template(
    "Write a short blog post using this outline:\n{outline}\n\nTopic: {topic}"
)

outline_chain = outline_prompt | llm | parser

article_chain = (
    RunnableParallel(
        outline=outline_chain,
        topic=RunnablePassthrough() | (lambda x: x["topic"]),
    )
    | article_prompt
    | llm
    | parser
)

result = article_chain.invoke({"topic": "Python decorators"})
print(result[:300], "...\n")

# ─── 2. Map-Reduce Chain ──────────────────────────────────────────────────────

print("=" * 55)
print("2. Map-Reduce: Summarize each doc, then combine")
print("=" * 55)

SAMPLE_DOCS = [
    "LangChain is a framework for building LLM applications. It supports chains, agents, and RAG.",
    "RAG stands for Retrieval-Augmented Generation. It combines vector search with LLM generation.",
    "LangGraph extends LangChain with stateful graph-based agent orchestration and cycle support.",
]

map_prompt = ChatPromptTemplate.from_template(
    "Summarize this text in one sentence:\n{doc}"
)
reduce_prompt = ChatPromptTemplate.from_template(
    "Combine these summaries into one coherent paragraph:\n{summaries}"
)

map_chain    = map_prompt    | llm | parser
reduce_chain = reduce_prompt | llm | parser

# Map: summarize each doc independently
summaries = map_chain.batch([{"doc": doc} for doc in SAMPLE_DOCS])
print("Individual summaries:")
for i, s in enumerate(summaries, 1):
    print(f"  {i}. {s}")

# Reduce: combine all summaries
combined = reduce_chain.invoke({"summaries": "\n".join(summaries)})
print(f"\nCombined:\n{combined}\n")

# ─── 3. Refine Chain (progressive refinement) ─────────────────────────────────

print("=" * 55)
print("3. Refine Chain: Build answer progressively")
print("=" * 55)

initial_prompt = ChatPromptTemplate.from_template(
    "Answer this question with the following context:\n{context}\n\nQuestion: {question}"
)
refine_prompt = ChatPromptTemplate.from_template(
    """Existing answer: {existing_answer}

New context to consider:
{context}

Refine the answer if the new context adds anything useful. Otherwise keep it unchanged."""
)

initial_chain = initial_prompt | llm | parser
refine_chain  = refine_prompt  | llm | parser

question = "What tools does LangChain provide?"
current_answer = initial_chain.invoke({"context": SAMPLE_DOCS[0], "question": question})
print(f"Initial: {current_answer}")

for doc in SAMPLE_DOCS[1:]:
    current_answer = refine_chain.invoke({
        "existing_answer": current_answer,
        "context": doc,
    })
    print(f"Refined: {current_answer}")

# ─── 4. Router Chain ──────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("4. Router Chain: Route to specialist")
print("=" * 55)

python_chain = (
    ChatPromptTemplate.from_template("You are a Python expert. Answer: {question}")
    | llm | parser
)
sql_chain = (
    ChatPromptTemplate.from_template("You are a SQL expert. Answer: {question}")
    | llm | parser
)
general_chain = (
    ChatPromptTemplate.from_template("Answer this question: {question}")
    | llm | parser
)

def detect_topic(x: dict) -> str:
    q = x["question"].lower()
    if any(w in q for w in ["python", "def ", "import", "list", "dict"]):
        return "python"
    elif any(w in q for w in ["sql", "select", "join", "table", "query"]):
        return "sql"
    return "general"

router = RunnableBranch(
    (lambda x: detect_topic(x) == "python", python_chain),
    (lambda x: detect_topic(x) == "sql",    sql_chain),
    general_chain,
)

test_questions = [
    {"question": "What is a Python list comprehension?"},
    {"question": "How do SQL JOIN types differ?"},
    {"question": "What is machine learning?"},
]

for q in test_questions:
    topic = detect_topic(q)
    answer = router.invoke(q)
    print(f"\n[{topic.upper()}] Q: {q['question']}")
    print(f"  A: {answer[:100]}...")
