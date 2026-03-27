# 17 — Callbacks & Tracing

## What are Callbacks?

Callbacks let you hook into the lifecycle of a LangChain run — getting notified when a chain starts, when the LLM generates tokens, when a tool is called, and when everything ends.

## Built-in Callbacks

```python
from langchain.callbacks import StdOutCallbackHandler

chain.invoke({"input": "Hello"}, config={"callbacks": [StdOutCallbackHandler()]})
```

## Custom Callback Handler

```python
from langchain_core.callbacks import BaseCallbackHandler

class MyLogger(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM started with {len(prompts)} prompts")

    def on_llm_end(self, response, **kwargs):
        print(f"LLM finished. Tokens: {response.llm_output}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"Tool '{serialized['name']}' called with: {input_str}")

    def on_chain_error(self, error, **kwargs):
        print(f"Chain error: {error}")

chain.invoke({"input": "Hello"}, config={"callbacks": [MyLogger()]})
```

## LangSmith Tracing

The best way to trace and debug LangChain apps. Set these env vars and everything is traced automatically:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=my-project
```

No code changes needed — every invoke, chain, and tool call is logged.

## Token Counting Callback

```python
from langchain_community.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = chain.invoke({"input": "Hello"})
    print(f"Tokens: {cb.total_tokens}, Cost: ${cb.total_cost:.4f}")
```

## Next Topic
→ [18 — LangSmith](../../05_production/18_langsmith/README.md)
