# 05 — LangChain Expression Language (LCEL)

## What is LCEL?

LCEL is LangChain's pipe-based composition system. It lets you chain together Runnables using the `|` operator — just like Unix pipes.

```python
# Old way (verbose)
prompt_value = prompt_template.format_prompt(**inputs)
ai_message   = llm.invoke(prompt_value)
output       = parser.parse(ai_message.content)

# LCEL way (elegant)
chain = prompt_template | llm | parser
output = chain.invoke(inputs)
```

Every component is a **Runnable** — it has the same interface regardless of what it does.

---

## The Runnable Interface

Every Runnable supports:

| Method | Description |
|--------|-------------|
| `.invoke(input)` | Call once, get one output |
| `.batch([input1, input2])` | Call multiple inputs in parallel |
| `.stream(input)` | Stream output token by token |
| `.ainvoke(input)` | Async version of invoke |
| `.abatch(inputs)` | Async batch |
| `.astream(input)` | Async stream |

---

## Key Runnables

### RunnablePassthrough
Passes input through unchanged. Useful for injecting the original input alongside transformed input.
```python
chain = RunnablePassthrough() | llm   # input passes straight to llm
```

### RunnableParallel
Runs multiple chains on the same input simultaneously.
```python
chain = RunnableParallel(
    summary=summarize_chain,
    translation=translate_chain,
)
# Returns: {"summary": "...", "translation": "..."}
```

### RunnableLambda
Wraps any Python function as a Runnable.
```python
double = RunnableLambda(lambda x: x * 2)
```

### RunnableBranch
Routes to different chains based on a condition.
```python
branch = RunnableBranch(
    (lambda x: "python" in x.lower(), python_chain),
    (lambda x: "java" in x.lower(),   java_chain),
    default_chain,  # fallback
)
```

---

## How LCEL Connects Components

```
Input dict
    ↓
ChatPromptTemplate  →  PromptValue (list of messages)
    ↓
ChatOpenAI          →  AIMessage
    ↓
StrOutputParser     →  str
    ↓
Output
```

Each component's output type must match the next component's expected input type.

---

## Benefits of LCEL

- **Streaming** — any LCEL chain streams automatically
- **Async** — every chain is async-ready
- **Batching** — parallel execution by default
- **Tracing** — auto-traced in LangSmith
- **Configurable** — swap components at runtime

---

## Next Topic

→ [06 — Chains](../06_chains/README.md)
