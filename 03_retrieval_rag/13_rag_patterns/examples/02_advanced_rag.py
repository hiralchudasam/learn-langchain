"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 13 — RAG Patterns                                     ║
║  File:   02_advanced_rag.py                                  ║
║  Level:  Advanced                                            ║
║  Goal:   HyDE, self-query, parent-document retrieval,        ║
║          and RAG evaluation metrics                          ║
╚══════════════════════════════════════════════════════════════╝
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

llm        = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Sample knowledge base
DOCS = [
    Document(page_content="LangChain was created by Harrison Chase in October 2022. It quickly became one of the fastest-growing open-source projects.", metadata={"source": "history.txt"}),
    Document(page_content="LCEL (LangChain Expression Language) was introduced in LangChain 0.1. It uses the pipe operator | to compose chains.", metadata={"source": "lcel.txt"}),
    Document(page_content="RAG stands for Retrieval-Augmented Generation. It combines vector search with LLM generation to answer questions about private data.", metadata={"source": "rag.txt"}),
    Document(page_content="LangGraph is a library for building stateful multi-agent applications. It was released in early 2024 as part of the LangChain ecosystem.", metadata={"source": "langgraph.txt"}),
    Document(page_content="LangSmith is the observability platform for LangChain. It provides tracing, evaluation, and monitoring capabilities.", metadata={"source": "langsmith.txt"}),
    Document(page_content="Vector databases store embeddings for semantic search. Popular options include Chroma, FAISS, Pinecone, and Weaviate.", metadata={"source": "vectorstores.txt"}),
]

splitter   = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks     = splitter.split_documents(DOCS)
vectorstore = Chroma.from_documents(chunks, embeddings, collection_name="advanced_rag_demo")
retriever  = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)

print("=" * 60)
print("13 — Advanced RAG Patterns")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# PATTERN 1: HyDE — Hypothetical Document Embeddings
# Problem: Short queries don't embed well → poor retrieval.
# Solution: Ask the LLM to generate a HYPOTHETICAL answer,
#           embed that instead — it's much closer to real docs.
#
# Short query:   "LangChain creator"       → weak retrieval
# HyDE query:    "LangChain was founded by Harrison Chase in 2022..."
#                                          → strong retrieval
# ─────────────────────────────────────────────────────────────
print("\n📌 1. HyDE — Hypothetical Document Embeddings")
print("─" * 40)

hyde_prompt = ChatPromptTemplate.from_template(
    "Write a short paragraph (2-3 sentences) that would answer this question:\n\n{question}"
)

def hyde_retrieve(question: str) -> list[Document]:
    """Generate a hypothetical answer, embed it, use for retrieval."""
    hyp_answer = (hyde_prompt | llm | StrOutputParser()).invoke({"question": question})
    print(f"  Hypothetical answer: {hyp_answer[:100]}...")
    return retriever.invoke(hyp_answer)  # embed the hypothesis, not the question

# Compare: direct retrieval vs HyDE
short_query = "who made langchain"

print(f"\n  Query: '{short_query}'")
print(f"\n  Direct retrieval:")
direct_docs = retriever.invoke(short_query)
for doc in direct_docs:
    print(f"    [{doc.metadata['source']}] {doc.page_content[:60]}")

print(f"\n  HyDE retrieval:")
hyde_docs = hyde_retrieve(short_query)
for doc in hyde_docs:
    print(f"    [{doc.metadata['source']}] {doc.page_content[:60]}")

# ─────────────────────────────────────────────────────────────
# PATTERN 2: Step-Back Prompting
# For specific/complex questions, first ask a more general question.
# The general answer provides better context for the specific answer.
# ─────────────────────────────────────────────────────────────
print("\n📌 2. Step-Back Prompting")
print("─" * 40)

stepback_prompt = ChatPromptTemplate.from_template(
    "Given this specific question, generate a more general, abstract question "
    "that would help answer it:\n\nSpecific: {question}\n\nGeneral:"
)

RAG_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "Use the context to answer. If unsure, say so.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

def step_back_rag(question: str) -> str:
    # Step 1: generate a broader question
    general_q = (stepback_prompt | llm | StrOutputParser()).invoke({"question": question})
    print(f"  Specific  : {question}")
    print(f"  Step-back : {general_q}")

    # Step 2: retrieve for both questions and combine
    specific_docs = retriever.invoke(question)
    general_docs  = retriever.invoke(general_q)
    all_docs = {d.page_content: d for d in specific_docs + general_docs}  # deduplicate
    context  = format_docs(list(all_docs.values()))

    # Step 3: answer
    return (RAG_TEMPLATE | llm | StrOutputParser()).invoke({"question": question, "context": context})

answer = step_back_rag("When exactly was the LCEL feature released in LangChain?")
print(f"  Answer    : {answer}")

# ─────────────────────────────────────────────────────────────
# PATTERN 3: Query Decomposition
# Break complex multi-part questions into sub-questions.
# Answer each, then synthesize.
# ─────────────────────────────────────────────────────────────
print("\n📌 3. Query Decomposition")
print("─" * 40)

decompose_prompt = ChatPromptTemplate.from_template(
    "Break this question into 2-3 simpler sub-questions. "
    "Return them as a numbered list:\n\n{question}"
)

def decompose_and_answer(question: str) -> str:
    # Decompose
    sub_questions_text = (decompose_prompt | llm | StrOutputParser()).invoke({"question": question})
    sub_questions = [
        line.split(". ", 1)[1].strip()
        for line in sub_questions_text.strip().split("\n")
        if line.strip() and line[0].isdigit()
    ]

    print(f"  Original: {question}")
    print(f"  Sub-questions:")

    # Answer each sub-question with RAG
    sub_answers = []
    for sq in sub_questions:
        docs   = retriever.invoke(sq)
        answer = (RAG_TEMPLATE | llm | StrOutputParser()).invoke({
            "question": sq, "context": format_docs(docs)
        })
        sub_answers.append(f"Q: {sq}\nA: {answer}")
        print(f"    → {sq}")
        print(f"       {answer[:80]}")

    # Synthesize
    synth_prompt = ChatPromptTemplate.from_template(
        "Using these Q&A pairs, write a comprehensive answer to: {original}\n\n{qa_pairs}"
    )
    return (synth_prompt | llm | StrOutputParser()).invoke({
        "original": question,
        "qa_pairs": "\n\n".join(sub_answers),
    })

final = decompose_and_answer("What is LangChain, when was it created, and what are its main components?")
print(f"\n  Final synthesized answer:\n  {final[:250]}")

# ─────────────────────────────────────────────────────────────
# PATTERN 4: RAG Evaluation — measure quality
# ─────────────────────────────────────────────────────────────
print("\n📌 4. Simple RAG Evaluation")
print("─" * 40)

from pydantic import BaseModel, Field

class RAGScore(BaseModel):
    faithfulness: int = Field(description="Is answer grounded in context? 1-5", ge=1, le=5)
    relevancy:    int = Field(description="Does answer address question? 1-5",   ge=1, le=5)

eval_llm = llm.with_structured_output(RAGScore)

def evaluate_rag_answer(question, context, answer):
    return (
        ChatPromptTemplate.from_messages([
            ("system", "You are a strict RAG evaluator."),
            ("human", f"Question: {question}\nContext: {context[:500]}\nAnswer: {answer}\nScore faithfulness and relevancy 1-5."),
        ]) | eval_llm
    ).invoke({})

# Build and evaluate a simple RAG chain
rag_chain = (
    RunnableParallel(context=(retriever | format_docs), question=RunnablePassthrough())
    | RAG_TEMPLATE | llm | StrOutputParser()
)

eval_questions = [
    "Who created LangChain?",
    "What is LangGraph used for?",
    "What is the weather today?",   # not in docs → should say IDK
]

print(f"  {'Question':<40} {'Faith'} {'Rel'}")
print(f"  {'─'*40} {'─'*5} {'─'*3}")
for q in eval_questions:
    answer  = rag_chain.invoke({"question": q})
    docs    = retriever.invoke(q)
    context = format_docs(docs)
    score   = evaluate_rag_answer(q, context, answer)
    print(f"  {q[:38]:<40} {score.faithfulness}/5   {score.relevancy}/5")

# ─────────────────────────────────────────────────────────────
# ⚠️  COMMON MISTAKE 1: Using the same query for retrieval
# The user's exact query may not match how docs are written.
# Use HyDE or MultiQuery to improve retrieval quality.

# ⚠️  COMMON MISTAKE 2: Blindly trusting retrieved context
# Retrieved docs can be irrelevant or contradictory.
# Always instruct the LLM: "If the answer is not in the context, say so."

# ⚠️  COMMON MISTAKE 3: Not evaluating retrieval separately
# A bad answer could come from bad retrieval OR bad generation.
# Evaluate them independently to know where to improve.
# ─────────────────────────────────────────────────────────────
print("\n⚠️  Advanced RAG tips:")
print("   • Use HyDE for short/vague queries")
print("   • Use step-back for very specific questions")
print("   • Use decomposition for multi-part questions")
print("   • Always evaluate faithfulness AND relevancy separately")
