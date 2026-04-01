"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 20 — Evaluation                                       ║
║  File:   02_eval_pipeline.py                                 ║
║  Level:  Advanced                                            ║
║  Goal:   Build a reusable evaluation pipeline, track         ║
║          scores over time, detect regressions                ║
╚══════════════════════════════════════════════════════════════╝
"""

from dotenv import load_dotenv
load_dotenv()

from dataclasses import dataclass, field
from typing import Callable, List
from datetime import datetime
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

print("=" * 60)
print("20 — Evaluation Pipeline")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# BUILDING BLOCK: Test case definition
# ─────────────────────────────────────────────────────────────
@dataclass
class TestCase:
    question:       str
    expected:       str           # expected keywords or answer
    category:       str = "general"

@dataclass
class EvalResult:
    question:       str
    expected:       str
    actual:         str
    score:          float         # 0.0 – 1.0
    passed:         bool
    category:       str
    latency_ms:     float

# ─────────────────────────────────────────────────────────────
# EVALUATORS: Different ways to score an answer
# ─────────────────────────────────────────────────────────────

def keyword_evaluator(answer: str, expected: str) -> float:
    """Score based on % of expected keywords found."""
    keywords = [w.strip().lower() for w in expected.split(",")]
    hits     = sum(1 for kw in keywords if kw in answer.lower())
    return hits / len(keywords)

class LLMScore(BaseModel):
    score:     float = Field(description="Score 0.0 to 1.0", ge=0, le=1)
    reasoning: str   = Field(description="One-sentence reason")

def llm_evaluator(question: str, answer: str, expected: str) -> float:
    """Use an LLM to grade correctness and relevance."""
    judge = llm.with_structured_output(LLMScore)
    result = (
        ChatPromptTemplate.from_messages([
            ("system", "Grade the answer 0.0–1.0 based on correctness."),
            ("human", f"Q: {question}\nExpected concepts: {expected}\nAnswer: {answer}"),
        ]) | judge
    ).invoke({})
    return result.score

# ─────────────────────────────────────────────────────────────
# EVALUATION RUNNER
# ─────────────────────────────────────────────────────────────

class EvalRunner:
    def __init__(self, chain, evaluator: Callable, pass_threshold: float = 0.6):
        self.chain          = chain
        self.evaluator      = evaluator
        self.pass_threshold = pass_threshold
        self.history:  List[List[EvalResult]] = []

    def run(self, test_cases: List[TestCase], label: str = "") -> List[EvalResult]:
        import time
        results = []
        for tc in test_cases:
            start  = time.time()
            actual = self.chain.invoke({"question": tc.question})
            ms     = (time.time() - start) * 1000
            score  = self.evaluator(tc.question, actual, tc.expected)
            results.append(EvalResult(
                question=tc.question, expected=tc.expected,
                actual=actual, score=round(score, 2),
                passed=score >= self.pass_threshold,
                category=tc.category, latency_ms=round(ms, 1),
            ))
        self.history.append(results)
        return results

    def report(self, results: List[EvalResult]):
        passed   = sum(1 for r in results if r.passed)
        avg_score = sum(r.score for r in results) / len(results)
        avg_ms    = sum(r.latency_ms for r in results) / len(results)

        print(f"\n  {'Question':<38} {'Score':<7} {'Pass':<6} {'ms'}")
        print(f"  {'─'*38} {'─'*7} {'─'*6} {'─'*6}")
        for r in results:
            status = "✅" if r.passed else "❌"
            print(f"  {r.question[:36]:<38} {r.score:<7.2f} {status:<6} {r.latency_ms:.0f}")

        print(f"\n  Passed    : {passed}/{len(results)} ({100*passed//len(results)}%)")
        print(f"  Avg score : {avg_score:.2f}")
        print(f"  Avg ms    : {avg_ms:.0f}ms")
        return {"pass_rate": passed/len(results), "avg_score": avg_score}

    def check_regression(self):
        """Alert if latest run is worse than previous."""
        if len(self.history) < 2:
            return
        prev_pass = sum(1 for r in self.history[-2] if r.passed) / len(self.history[-2])
        curr_pass = sum(1 for r in self.history[-1] if r.passed) / len(self.history[-1])
        delta     = curr_pass - prev_pass
        if delta < -0.1:
            print(f"\n  ⚠️  REGRESSION DETECTED! Pass rate dropped {delta*100:.0f}%")
        elif delta > 0.05:
            print(f"\n  ✅  Improvement! Pass rate up {delta*100:.0f}%")
        else:
            print(f"\n  ─  No significant change ({delta*100:+.0f}%)")

# ─────────────────────────────────────────────────────────────
# RUN THE EVAL
# ─────────────────────────────────────────────────────────────
qa_chain = (
    ChatPromptTemplate.from_template("Answer briefly: {question}")
    | llm | parser
)

TEST_SUITE = [
    TestCase("What is LangChain?",          "framework,LLM,applications",    "langchain"),
    TestCase("What is RAG?",                "retrieval,generation,documents", "rag"),
    TestCase("What does LCEL stand for?",   "LangChain,Expression,Language",  "lcel"),
    TestCase("What is the capital of India?","New Delhi,Delhi",               "geography"),
    TestCase("What is 7 * 8?",              "56",                             "math"),
    TestCase("Who created LangChain?",      "Harrison Chase",                 "history"),
]

runner = EvalRunner(qa_chain, llm_evaluator, pass_threshold=0.6)

print("\n📌 1. First Evaluation Run (baseline)")
results_v1 = runner.run(TEST_SUITE, label="v1-baseline")
summary_v1 = runner.report(results_v1)

# Simulate a second run (e.g. after a prompt change)
print("\n📌 2. Second Evaluation Run (after prompt change)")
qa_chain_v2 = (
    ChatPromptTemplate.from_template(
        "You are an expert assistant. Answer concisely and accurately: {question}"
    )
    | llm | parser
)
runner.chain = qa_chain_v2
results_v2 = runner.run(TEST_SUITE, label="v2-improved-prompt")
summary_v2 = runner.report(results_v2)

runner.check_regression()

# ─────────────────────────────────────────────────────────────
# CATEGORY BREAKDOWN
# ─────────────────────────────────────────────────────────────
print("\n📌 3. Results by Category")
print("─" * 40)

from collections import defaultdict
category_scores = defaultdict(list)
for r in results_v2:
    category_scores[r.category].append(r.score)

for cat, scores in sorted(category_scores.items()):
    avg = sum(scores) / len(scores)
    bar = "█" * int(avg * 10)
    print(f"  {cat:<12} {bar:<12} {avg:.2f}")

print("\n⚠️  Eval pipeline tips:")
print("   • Always save a baseline before making changes")
print("   • Run evals per category to spot weak areas")
print("   • Use LLM evaluator for nuanced grading")
print("   • Use keyword evaluator for fast/cheap CI checks")
