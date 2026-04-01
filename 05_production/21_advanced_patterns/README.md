# 21 — Advanced Patterns

## Caching

### In-Memory Cache (per session)
```python
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
set_llm_cache(InMemoryCache())
```

### SQLite Cache (persistent across runs)
```python
from langchain.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))
```

## Streaming + Async

```python
async def stream_response(query: str):
    async for chunk in chain.astream({"input": query}):
        yield chunk

# In FastAPI
@app.get("/stream")
async def stream(q: str):
    from fastapi.responses import StreamingResponse
    return StreamingResponse(stream_response(q), media_type="text/plain")
```

## Guardrails

```python
from langchain_core.runnables import RunnableLambda

def check_safe(input_dict: dict) -> dict:
    text = input_dict.get("input", "")
    blocked = ["hack", "exploit", "illegal"]
    if any(word in text.lower() for word in blocked):
        raise ValueError("Blocked: unsafe content detected")
    return input_dict

safe_chain = RunnableLambda(check_safe) | my_chain
```

## Fallbacks

```python
# If gpt-4o fails, fall back to gpt-4o-mini
robust_llm = ChatOpenAI(model="gpt-4o").with_fallbacks([
    ChatOpenAI(model="gpt-4o-mini")
])
```

## Retry Logic

```python
from langchain_core.runnables import RunnableRetry

chain_with_retry = my_chain.with_retry(
    retry_if_exception_type=(Exception,),
    stop_after_attempt=3,
)
```

## Cost Optimization

```python
# 1. Cache common queries
# 2. Use smaller models for simple tasks
# 3. Compress prompts (remove whitespace, shorten system prompts)
# 4. Reduce chunk size for embeddings
# 5. Use batching instead of sequential calls
# 6. Use streaming to improve perceived latency

# Token counting before sending
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o-mini")
token_count = len(enc.encode(prompt))
if token_count > 4000:
    # summarize or truncate first
    pass
```

## Multi-Agent Pattern

```python
# Supervisor routes to specialist agents
supervisor = create_react_agent(llm, [route_to_researcher, route_to_coder])

# Specialist agents
researcher = create_react_agent(llm, [web_search, summarize])
coder      = create_react_agent(llm, [python_repl, write_file])
```

---

*You've completed the curriculum! Move on to the Projects section.* 🎉
