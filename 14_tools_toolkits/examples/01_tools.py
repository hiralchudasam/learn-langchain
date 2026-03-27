"""
14 — Tools & Toolkits
Example 01: Creating tools, binding to LLM, executing tool calls
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field
from typing import Optional
import json
import random

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ─── 1. Basic @tool decorator ─────────────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Returns temperature and conditions."""
    mock_data = {
        "mumbai":    "34°C, hot and humid",
        "delhi":     "28°C, sunny and clear",
        "bangalore": "24°C, pleasant with light breeze",
        "chennai":   "36°C, very hot",
        "kolkata":   "30°C, partly cloudy",
    }
    city_lower = city.lower()
    return mock_data.get(city_lower, f"Weather data unavailable for {city}")

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Example: '2 + 2 * 10'."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"

@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol like AAPL, GOOGL, MSFT."""
    mock_prices = {
        "AAPL": 189.45, "GOOGL": 175.23, "MSFT": 412.87,
        "TSLA": 248.92, "AMZN": 198.56,
    }
    ticker_upper = ticker.upper()
    price = mock_prices.get(ticker_upper, round(random.uniform(50, 500), 2))
    return f"{ticker_upper}: ${price}"

print("Tool name:", get_weather.name)
print("Tool desc:", get_weather.description)
print("Tool schema:", get_weather.args_schema.model_json_schema())

# ─── 2. StructuredTool (multiple inputs with schema) ──────────────────────────

class ConvertInput(BaseModel):
    amount: float = Field(description="Amount to convert")
    from_currency: str = Field(description="Source currency code e.g. USD")
    to_currency: str = Field(description="Target currency code e.g. INR")

def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Converts currency using mock exchange rates."""
    rates = {"USD": 1.0, "INR": 83.5, "EUR": 0.92, "GBP": 0.79, "JPY": 149.5}
    if from_currency not in rates or to_currency not in rates:
        return "Currency not supported"
    result = amount * (rates[to_currency] / rates[from_currency])
    return f"{amount} {from_currency} = {result:.2f} {to_currency}"

currency_tool = StructuredTool.from_function(
    func=convert_currency,
    name="convert_currency",
    description="Convert an amount from one currency to another",
    args_schema=ConvertInput,
)

print("\nStructuredTool test:", currency_tool.invoke({"amount": 100, "from_currency": "USD", "to_currency": "INR"}))

# ─── 3. Bind Tools to LLM ─────────────────────────────────────────────────────

tools = [get_weather, calculate, get_stock_price, currency_tool]
llm_with_tools = llm.bind_tools(tools)

print("\n" + "=" * 50)
print("3. LLM deciding which tool to call")
print("=" * 50)

queries = [
    "What is the weather in Mumbai?",
    "What is 1234 * 5678?",
    "What is the stock price of AAPL?",
    "Convert 500 USD to INR",
    "What is the capital of France?",  # No tool needed
]

for query in queries:
    response = llm_with_tools.invoke([HumanMessage(content=query)])
    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"\nQ: {query}")
            print(f"   → Tool: {tc['name']}({tc['args']})")
    else:
        print(f"\nQ: {query}")
        print(f"   → Direct: {response.content}")

# ─── 4. Full Tool Execution Loop ──────────────────────────────────────────────

print("\n" + "=" * 50)
print("4. Full tool execution loop")
print("=" * 50)

tools_by_name = {t.name: t for t in tools}

def run_agent_step(messages: list) -> str:
    """Run one complete agent step: LLM → tool calls → final answer."""
    while True:
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        # Execute each tool call
        for tc in response.tool_calls:
            tool_fn = tools_by_name[tc["name"]]
            result = tool_fn.invoke(tc["args"])
            print(f"  [Tool] {tc['name']}({tc['args']}) → {result}")
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"])
            )

complex_query = "What's the weather in Delhi and Mumbai, and convert 200 USD to INR?"
print(f"\nQuery: {complex_query}")
answer = run_agent_step([HumanMessage(content=complex_query)])
print(f"Answer: {answer}")
