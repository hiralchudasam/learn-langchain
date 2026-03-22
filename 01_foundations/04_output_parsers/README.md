# 04 — Output Parsers

## What are Output Parsers?

By default, a ChatModel returns an `AIMessage` object. Output parsers transform that into a more useful Python type — a plain string, a list, a dict, or a typed Pydantic object.

```
ChatModel → AIMessage → OutputParser → str / list / dict / Pydantic
```

---

## Common Output Parsers

### StrOutputParser
Extracts the text content from an AIMessage. The most commonly used parser.
```python
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()
# AIMessage("Paris") → "Paris"
```

### JsonOutputParser
Parses JSON from the model's response.
```python
from langchain_core.output_parsers import JsonOutputParser
# Requires prompting the model to return valid JSON
```

### PydanticOutputParser
Parses the output into a typed Pydantic model. Gives you full type safety.
```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

class Movie(BaseModel):
    title: str
    year: int
    genre: str

parser = PydanticOutputParser(pydantic_object=Movie)
```

### CommaSeparatedListOutputParser
Splits a comma-separated string into a Python list.
```python
from langchain.output_parsers import CommaSeparatedListOutputParser
# "red, green, blue" → ["red", "green", "blue"]
```

---

## Format Instructions

Pydantic and JSON parsers provide `format_instructions` to tell the LLM exactly what format to use:

```python
parser = PydanticOutputParser(pydantic_object=Movie)
print(parser.get_format_instructions())
# "Return a JSON object with fields: title (str), year (int), genre (str)"
```

Always inject format instructions into your prompt:
```python
template = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's question.\n{format_instructions}"),
    ("human", "{question}"),
])
chain = template.partial(format_instructions=parser.get_format_instructions()) | llm | parser
```

---

## With Structured Output (Recommended Modern Approach)

LangChain 0.3+ supports `.with_structured_output()` — no manual parser needed:

```python
llm_structured = llm.with_structured_output(Movie)
result = llm_structured.invoke("Tell me about Inception")
# result is a Movie(title="Inception", year=2010, genre="Sci-Fi")
```

---

## Error Handling

Use `OutputFixingParser` to auto-retry if parsing fails:
```python
from langchain.output_parsers import OutputFixingParser
robust_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
```

---

## Next Topic

→ [05 — LCEL](../../02_core_abstractions/05_lcel/README.md)
