# 03 — Prompt Templates

## What are Prompt Templates?

A prompt template is a reusable, parameterized prompt. Instead of hard-coding your prompt every time, you define a template with placeholders and fill them in at runtime.

```python
# Without template (bad)
prompt = f"Translate '{user_text}' to {language}"

# With template (good)
template = PromptTemplate.from_template("Translate '{text}' to {language}")
prompt = template.invoke({"text": user_text, "language": language})
```

---

## Types of Prompt Templates

### PromptTemplate
For simple string-based prompts.
```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template("Tell me a joke about {topic}")
prompt = template.invoke({"topic": "Python"})
# → StringPromptValue(text="Tell me a joke about Python")
```

### ChatPromptTemplate
For chat-based models (the modern standard). Supports system + human + ai messages.
```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who speaks like {persona}."),
    ("human", "{question}"),
])
messages = template.invoke({"persona": "Shakespeare", "question": "What is AI?"})
```

### FewShotPromptTemplate
Includes examples to guide the model's output style.

### MessagesPlaceholder
Inserts a list of messages (e.g., conversation history) into a template.
```python
from langchain_core.prompts import MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),   # ← inject chat history here
    ("human", "{input}"),
])
```

---

## Partial Templates

Fill in only some variables now, the rest later:

```python
template = PromptTemplate.from_template("Translate '{text}' to {language}")
spanish_template = template.partial(language="Spanish")

# Later:
prompt = spanish_template.invoke({"text": "Hello"})
```

---

## Composition with LCEL

Prompt templates are Runnables and can be piped directly into a model:

```python
chain = template | llm | StrOutputParser()
result = chain.invoke({"topic": "Python"})
```

---

## Inspecting a Template

```python
template.input_variables    # ['topic', 'language']
template.input_schema       # Pydantic schema
template.format(**inputs)   # Returns the formatted string
```

---

## Best Practices

- Always use templates instead of f-strings — they validate inputs
- Use `ChatPromptTemplate` for ChatModels, `PromptTemplate` for plain LLMs
- Use `MessagesPlaceholder` to inject conversation history
- Keep system prompts concise and role-specific

---

## Next Topic

→ [04 — Output Parsers](../04_output_parsers/README.md)
