# 20 — Evaluation & Testing

## Why Evaluate LLM Apps?

LLMs are non-deterministic — the same input can produce different outputs. Evaluation helps you:
- Measure quality before and after changes
- Catch regressions when you change prompts or models
- Quantify improvements objectively

## LangSmith Evaluators

```python
from langsmith.evaluation import evaluate, LangChainStringEvaluator

# Correctness evaluator (compares to reference answer)
evaluator = LangChainStringEvaluator("qa", config={"llm": llm})

results = evaluate(
    lambda x: my_chain.invoke(x),
    data="my-dataset-name",
    evaluators=[evaluator],
    experiment_prefix="gpt4o-mini-test",
)
```

## Custom Evaluator

```python
from langsmith.schemas import Run, Example
from langsmith.evaluation import evaluate

def contains_citation(run: Run, example: Example) -> dict:
    output = run.outputs.get("output", "")
    has_citation = "source:" in output.lower() or "[" in output
    return {"key": "has_citation", "score": int(has_citation)}

results = evaluate(my_chain, data="rag-test-set", evaluators=[contains_citation])
```

## RAGAS (RAG Evaluation)

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

result = evaluate(
    dataset=ragas_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
)
print(result.to_pandas())
```

## Key RAG Metrics

| Metric | Question | Good Score |
|--------|---------|-----------|
| Faithfulness | Is the answer grounded in the context? | > 0.8 |
| Answer relevancy | Does the answer address the question? | > 0.8 |
| Context precision | Are retrieved docs actually relevant? | > 0.7 |
| Context recall | Did we retrieve all relevant docs? | > 0.7 |

## Next Topic
→ [21 — Advanced Patterns](../21_advanced_patterns/README.md)
