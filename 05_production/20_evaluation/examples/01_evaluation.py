"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 20 — Evaluation & Testing                             ║
║  File:   01_evaluation.py                                    ║
║  Level:  Basic → Advanced                                    ║
║  Goal:   Evaluate LLM outputs using criteria evaluators,     ║
║          custom metrics, and RAG-specific evaluation         ║
╚══════════════════════════════════════════════════════════════╝
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

print("=" * 60)
print("20 — Evaluating LLM Outputs")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# SECTION 1: Why evaluate LLMs?
# LLMs are non-deterministic — "looks right" isn't good enough.
# You need automated checks to:
#   - Catch regressions when you change prompts/models
#   - Measure quality across a test suite
#   - Compare model A vs model B
# ─────────────────────────────────────────────────────────────
print("\n📌 1. LLM-as-Judge — Using an LLM to Evaluate Output")
print("─" * 40)

# The simplest evaluator: ask an LLM to grade the output
class EvalResult(BaseModel):
    score:      int  = Field(description="Score from 1-5", ge=1, le=5)
    reasoning:  str  = Field(description="Brief reasoning for the score")
    passed:     bool = Field(description="True if score >= 3")

eval_llm = llm.with_structured_output(EvalResult)

def evaluate_answer(question: str, answer: str, criteria: str) -> EvalResult:
    """Use an LLM to grade an answer based on given criteria."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict evaluator. Grade the answer honestly."),
        ("human",
         f"Question : {question}\n"
         f"Answer   : {answer}\n"
         f"Criteria : {criteria}\n"
         f"Rate 1-5 (5=perfect, 1=completely wrong)."),
    ])
    return (prompt | eval_llm).invoke({})

# Chain we want to evaluate
qa_chain = (
    ChatPromptTemplate.from_template("Answer briefly: {question}")
    | llm | parser
)

# Test cases
test_cases = [
    {
        "question": "What is the capital of France?",
        "criteria": "Correctness — must say Paris",
    },
    {
        "question": "Explain gradient descent in simple terms.",
        "criteria": "Clarity and accuracy for a beginner",
    },
    {
        "question": "What is 2 + 2?",
        "criteria": "Mathematical correctness",
    },
]

print(f"  {'Question':<40} {'Score':<7} {'Passed'}")
print(f"  {'─'*40} {'─'*7} {'─'*6}")

for tc in test_cases:
    answer = qa_chain.invoke({"question": tc["question"]})
    result = evaluate_answer(tc["question"], answer, tc["criteria"])
    status = "✅" if result.passed else "❌"
    print(f"  {tc['question'][:38]:<40} {result.score}/5    {status}")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Criteria-Based Evaluation
# Pre-defined criteria: conciseness, relevance, harmfulness, etc.
# ─────────────────────────────────────────────────────────────
print("\n📌 2. Multi-Criteria Evaluation")
print("─" * 40)

CRITERIA = {
    "relevance":    "Does the answer actually address the question?",
    "conciseness":  "Is the answer appropriately brief without losing key info?",
    "accuracy":     "Is the factual content correct?",
    "clarity":      "Is the answer easy to understand?",
}

class CriteriaScore(BaseModel):
    criterion:  str  = Field(description="Which criterion is being scored")
    score:      int  = Field(description="Score 1-5", ge=1, le=5)
    reasoning:  str  = Field(description="One-sentence reasoning")

def multi_criteria_eval(question: str, answer: str) -> list[CriteriaScore]:
    """Evaluate an answer across multiple criteria."""
    results = []
    for criterion, definition in CRITERIA.items():
        eval_result = evaluate_answer(question, answer, f"{criterion}: {definition}")
        results.append(CriteriaScore(
            criterion=criterion,
            score=eval_result.score,
            reasoning=eval_result.reasoning[:60],
        ))
    return results

question = "What is machine learning?"
answer   = qa_chain.invoke({"question": question})
print(f"  Question: {question}")
print(f"  Answer  : {answer[:100]}...\n")

scores    = multi_criteria_eval(question, answer)
avg_score = sum(s.score for s in scores) / len(scores)

for s in scores:
    bar = "█" * s.score + "░" * (5 - s.score)
    print(f"  {s.criterion:<15} {bar} {s.score}/5  {s.reasoning[:50]}")

print(f"\n  Average score: {avg_score:.1f}/5.0")

# ─────────────────────────────────────────────────────────────
# SECTION 3: RAG-Specific Evaluation
# For RAG pipelines, you need to evaluate THREE things:
#   1. Context precision — were the right docs retrieved?
#   2. Faithfulness — is the answer grounded in the context?
#   3. Answer relevancy — does the answer address the question?
# ─────────────────────────────────────────────────────────────
print("\n📌 3. RAG Pipeline Evaluation")
print("─" * 40)

class RAGEval(BaseModel):
    faithfulness:      int = Field(description="Is the answer grounded in context? (1-5)", ge=1, le=5)
    answer_relevancy:  int = Field(description="Does answer address the question? (1-5)", ge=1, le=5)
    context_precision: int = Field(description="Is the context relevant to the question? (1-5)", ge=1, le=5)
    hallucination:     bool = Field(description="Does the answer contain info NOT in context?")
    reasoning:         str  = Field(description="Overall assessment in one sentence")

rag_eval_llm = llm.with_structured_output(RAGEval)

def evaluate_rag(question: str, context: str, answer: str) -> RAGEval:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert RAG evaluator. Be strict and honest."),
        ("human",
         f"QUESTION : {question}\n\n"
         f"CONTEXT  : {context}\n\n"
         f"ANSWER   : {answer}\n\n"
         f"Evaluate faithfulness, answer relevancy, context precision, "
         f"and whether the answer contains hallucinations."),
    ])
    return (prompt | rag_eval_llm).invoke({})

# Simulate RAG scenarios
rag_test_cases = [
    {
        "question": "What is LangChain?",
        "context":  "LangChain is an open-source framework for building LLM applications. "
                    "It was created by Harrison Chase in 2022.",
        "answer":   "LangChain is a framework for LLM apps, created by Harrison Chase in 2022.",
    },
    {
        "question": "Who created LangChain?",
        "context":  "LangChain supports chains, agents, and retrieval strategies.",
        "answer":   "LangChain was created by Sam Altman.",   # hallucination!
    },
]

for i, tc in enumerate(rag_test_cases, 1):
    result = evaluate_rag(tc["question"], tc["context"], tc["answer"])
    print(f"\n  Test {i}: '{tc['question']}'")
    print(f"  Answer     : {tc['answer'][:70]}")
    print(f"  Faithfulness    : {result.faithfulness}/5")
    print(f"  Ans Relevancy   : {result.answer_relevancy}/5")
    print(f"  Ctx Precision   : {result.context_precision}/5")
    print(f"  Hallucination   : {'⚠️ YES' if result.hallucination else '✅ NO'}")
    print(f"  Assessment      : {result.reasoning}")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Batch Evaluation (test suite)
# Run your chain against a set of test cases and score them.
# ─────────────────────────────────────────────────────────────
print("\n📌 4. Running a Test Suite")
print("─" * 40)

test_suite = [
    {"question": "What is Python?",          "expected": "programming language"},
    {"question": "What is 10 + 5?",          "expected": "15"},
    {"question": "What is the speed of light?", "expected": "299,792,458"},
    {"question": "Who invented the telephone?", "expected": "Alexander Graham Bell"},
]

def contains_expected(answer: str, expected: str) -> bool:
    """Simple keyword-based check."""
    return any(word.lower() in answer.lower() for word in expected.split())

passed = 0
for tc in test_suite:
    answer = qa_chain.invoke({"question": tc["question"]})
    ok     = contains_expected(answer, tc["expected"])
    if ok:
        passed += 1
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"  {status} | Q: {tc['question'][:35]:<35} | A: {answer[:40]}")

print(f"\n  Score: {passed}/{len(test_suite)} ({100*passed//len(test_suite)}%)")

# ─────────────────────────────────────────────────────────────
# ⚠️  COMMON MISTAKE 1: Evaluating with the same model
# Using gpt-4o-mini to evaluate gpt-4o-mini responses is biased.
# Use a stronger model (gpt-4o) as the judge when possible.

# ⚠️  COMMON MISTAKE 2: Only evaluating the final answer
# For RAG, you must also evaluate retrieval quality separately.
# A perfect answer from bad retrieval is a fluke, not reliable.

# ⚠️  COMMON MISTAKE 3: Not testing edge cases
# Always include: empty input, very long input, ambiguous questions,
# and questions where the answer is "I don't know".
# ─────────────────────────────────────────────────────────────
print("\n⚠️  Evaluation tips:")
print("   • Use a stronger model as judge (avoid self-evaluation bias)")
print("   • Evaluate RAG retrieval AND generation separately")
print("   • Include edge cases in your test suite")
print("   • Run evals before AND after changing prompts/models")
