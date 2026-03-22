# 02 — Language Models

## Types of Models in LangChain

LangChain supports three types of model interfaces:

### 1. LLM (Text In → Text Out)
Takes a plain string and returns a plain string. Older style.
```python
llm.invoke("What is the capital of France?")  # → "Paris"
```

### 2. ChatModel (Messages In → Message Out)
Takes a list of messages (system, human, ai) and returns an AI message.  
This is the **modern standard** — use this for everything.
```python
chat.invoke([HumanMessage("What is the capital of France?")])  # → AIMessage("Paris")
```

### 3. Embedding Model (Text In → Vector Out)
Converts text into a numerical vector for semantic search.
```python
embeddings.embed_query("What is RAG?")  # → [0.021, -0.034, ...]
```

---

## Supported Providers

| Provider | Package | Models |
|----------|---------|--------|
| OpenAI | `langchain-openai` | gpt-4o, gpt-4o-mini, gpt-3.5-turbo |
| Anthropic | `langchain-anthropic` | claude-3-5-sonnet, claude-3-haiku |
| Google | `langchain-google-genai` | gemini-1.5-pro, gemini-flash |
| HuggingFace | `langchain-huggingface` | Any HF model |
| Ollama | `langchain-ollama` | llama3, mistral, phi3 (local) |
| Groq | `langchain-groq` | llama3, mixtral (fast inference) |

---

## Key Parameters

| Parameter | Type | Effect |
|-----------|------|--------|
| `temperature` | 0.0–2.0 | 0 = deterministic, 2 = very random |
| `max_tokens` | int | Max output length |
| `model` | str | Which model to use |
| `streaming` | bool | Stream tokens as they generate |

---

## Message Types

```python
from langchain_core.messages import (
    SystemMessage,    # Sets the assistant's persona/instructions
    HumanMessage,     # User's input
    AIMessage,        # Model's response
    ToolMessage,      # Result from a tool call
)
```

---

## Response Object

When you call `.invoke()` on a ChatModel, you get back an `AIMessage`:

```python
response = chat.invoke([HumanMessage("Hello")])

response.content          # The text: "Hello! How can I help?"
response.usage_metadata   # Token counts: {'input_tokens': 9, 'output_tokens': 7}
response.response_metadata  # Model name, finish reason, etc.
```

---

## Streaming

```python
for chunk in chat.stream([HumanMessage("Tell me a story")]):
    print(chunk.content, end="", flush=True)
```

---

## Next Topic

→ [03 — Prompt Templates](../03_prompt_templates/README.md)
