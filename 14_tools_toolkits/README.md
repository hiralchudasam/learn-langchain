# 14 — Tools & Toolkits

## What is a Tool?

A tool is a function that an LLM agent can call. The LLM decides *when* and *which* tool to invoke based on the user's query.

```
User: "What is the weather in Mumbai?"
LLM: I need the weather tool → calls get_weather("Mumbai") → gets result → answers user
```

---

## Creating Tools

### Method 1: @tool decorator (simplest)
```python
from langchain_core.tools import tool

@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol."""
    # implementation
    return f"{ticker}: $150.23"
```

The docstring becomes the tool's description — the LLM reads it to decide when to use it.

### Method 2: StructuredTool (for complex input schemas)
```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

class WeatherInput(BaseModel):
    city: str
    unit: str = "celsius"

weather_tool = StructuredTool.from_function(
    func=get_weather,
    name="get_weather",
    description="Get weather for a city",
    args_schema=WeatherInput,
)
```

### Method 3: Tool class
```python
from langchain_core.tools import Tool

tool = Tool(
    name="calculator",
    func=lambda x: eval(x),
    description="Evaluates a math expression string",
)
```

---

## Built-in Tools

```python
# Web search
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults(max_results=3)

# Wikipedia
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Python REPL
from langchain_experimental.tools import PythonREPLTool
repl = PythonREPLTool()

# DuckDuckGo (free, no API key)
from langchain_community.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()
```

---

## Toolkits

A **toolkit** is a group of related tools for a specific service.

```python
# SQL Toolkit
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///mydb.db")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# File System Toolkit
from langchain_community.agent_toolkits import FileManagementToolkit
toolkit = FileManagementToolkit(root_dir="/tmp/agent_workspace")
```

---

## Binding Tools to LLM

```python
llm_with_tools = llm.bind_tools([search, wiki, calculator])
response = llm_with_tools.invoke("What is the population of India?")

# Check if the model wants to call a tool
if response.tool_calls:
    for tc in response.tool_calls:
        print(tc["name"], tc["args"])
```

---

## Tool Best Practices

- Write clear, specific docstrings — the LLM reads them
- Return strings or JSON-serializable types
- Handle errors gracefully inside the tool
- Keep tools focused — one tool, one job
- Add type hints for better schema generation

---

## Next Topic

→ [15 — Agents](../15_agents/README.md)
