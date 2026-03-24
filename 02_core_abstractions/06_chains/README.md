# 06 — Chains

## What are Chains?

A chain is a sequence of calls — to LLMs, prompts, parsers, tools, or other chains. With LCEL, you build chains by piping Runnables together using `|`.

## Common Chain Patterns

### Simple LLM Chain
```python
chain = prompt | llm | parser
```

### Sequential Chain (output of one → input of next)
```python
outline_chain = prompt1 | llm | StrOutputParser()
article_chain = prompt2 | llm | StrOutputParser()

full_chain = (
    {"outline": outline_chain, "topic": RunnablePassthrough()}
    | article_chain
)
```

### Map-Reduce Chain
Process each document separately, then combine:
```python
map_chain   = map_prompt   | llm | StrOutputParser()
reduce_chain = reduce_prompt | llm | StrOutputParser()

def map_reduce(docs):
    summaries = map_chain.batch([{"doc": d.page_content} for d in docs])
    return reduce_chain.invoke({"summaries": "\n".join(summaries)})
```

### Router Chain
Send input to different chains based on content:
```python
branch = RunnableBranch(
    (lambda x: x["topic"] == "python", python_chain),
    (lambda x: x["topic"] == "sql",    sql_chain),
    general_chain,
)
```

## Legacy Chains (v0.1)
LangChain has legacy classes like `LLMChain`, `SequentialChain`, `RetrievalQA`. These still work but the LCEL approach above is preferred.

## Next Topic
→ [07 — Memory](../07_memory/README.md)
