"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 14 — Tools & Toolkits                                 ║
║  File:   02_tools_advanced.py                                ║
║  Level:  Advanced                                            ║
║  Goal:   Error handling in tools, async tools, toolkits,     ║
║          tool injection, and tool best practices             ║
╚══════════════════════════════════════════════════════════════╝
"""

from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, StructuredTool, InjectedToolArg
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field
from typing import Annotated
from datetime import datetime

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("=" * 60)
print("14 — Advanced Tools")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# SECTION 1: Tools with Error Handling
# Tools should NEVER raise exceptions — they should return
# an error message string so the agent can recover gracefully.
# ─────────────────────────────────────────────────────────────
print("\n📌 1. Tools with Error Handling")
print("─" * 40)

@tool
def safe_divide(a: float, b: float) -> str:
    """Divide a by b. Returns the result or an error message."""
    # ✅ Return error as string — agent can read and recover
    if b == 0:
        return "Error: Cannot divide by zero. Please provide a non-zero divisor."
    return str(round(a / b, 4))

@tool
def lookup_user(user_id: int) -> str:
    """Look up user info by their numeric ID."""
    users = {1: "Alice (alice@example.com)", 2: "Bob (bob@example.com)"}
    if user_id not in users:
        return f"Error: No user found with ID {user_id}. Valid IDs are: {list(users.keys())}"
    return users[user_id]

print(f"  safe_divide(10, 2)   = {safe_divide.invoke({'a': 10, 'b': 2})}")
print(f"  safe_divide(10, 0)   = {safe_divide.invoke({'a': 10, 'b': 0})}")
print(f"  lookup_user(1)       = {lookup_user.invoke({'user_id': 1})}")
print(f"  lookup_user(999)     = {lookup_user.invoke({'user_id': 999})}")
print(f"  ← Both tools return error strings — agent can handle gracefully")

# ─────────────────────────────────────────────────────────────
# SECTION 2: handle_tool_error in bind_tools
# An alternative: LangChain can catch exceptions automatically
# and convert them to tool error messages for the agent.
# ─────────────────────────────────────────────────────────────
print("\n📌 2. Automatic Tool Error Handling")
print("─" * 40)

@tool
def risky_tool(query: str) -> str:
    """A tool that might raise an exception."""
    if "fail" in query.lower():
        raise ValueError(f"Tool failed for query: '{query}'")
    return f"Success: processed '{query}'"

# handle_tool_error=True → LangChain catches exceptions,
# returns them as tool error messages instead of crashing
llm_safe = llm.bind_tools([risky_tool], handle_tool_errors=True)

print(f"  risky_tool('hello') = {risky_tool.invoke({'query': 'hello'})}")
try:
    result = risky_tool.invoke({"query": "please fail"})
except Exception as e:
    print(f"  risky_tool('fail')  = Exception: {e}")
    print(f"  ← Use handle_tool_errors=True in bind_tools to auto-catch these")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Async Tools
# For tools that do I/O (API calls, DB queries) — use async
# so the agent doesn't block while waiting.
# ─────────────────────────────────────────────────────────────
print("\n📌 3. Async Tools")
print("─" * 40)

import asyncio

@tool
async def fetch_exchange_rate(from_currency: str, to_currency: str) -> str:
    """Fetch the exchange rate between two currencies (async mock)."""
    await asyncio.sleep(0.01)  # simulate I/O
    rates = {
        ("USD", "INR"): 83.5,
        ("USD", "EUR"): 0.92,
        ("EUR", "INR"): 90.8,
    }
    key = (from_currency.upper(), to_currency.upper())
    rate = rates.get(key, rates.get((key[1], key[0])))
    if rate is None:
        return f"Exchange rate for {from_currency}/{to_currency} not available."
    return f"1 {from_currency.upper()} = {rate} {to_currency.upper()}"

# Async tools can be called with ainvoke
async def demo_async():
    result = await fetch_exchange_rate.ainvoke({"from_currency": "USD", "to_currency": "INR"})
    print(f"  USD/INR rate: {result}")

    # Or invoke synchronously (LangChain handles the event loop)
    result2 = fetch_exchange_rate.invoke({"from_currency": "EUR", "to_currency": "INR"})
    print(f"  EUR/INR rate: {result2}")

asyncio.run(demo_async())

# ─────────────────────────────────────────────────────────────
# SECTION 4: InjectedToolArg — hidden parameters
# Some tool args should be injected by your code (like a DB session,
# API client, or user ID) — NOT asked of the LLM.
# Mark them with InjectedToolArg so the LLM doesn't see them.
# ─────────────────────────────────────────────────────────────
print("\n📌 4. InjectedToolArg — Hidden Parameters")
print("─" * 40)

# Simulated database
USER_DB = {
    "alice": {"balance": 5000, "plan": "premium"},
    "bob":   {"balance": 1500, "plan": "basic"},
}

@tool
def get_account_info(
    query: str,
    # ← LLM provides this
    user_id: Annotated[str, InjectedToolArg],
    # ← your code injects this, LLM never sees it
) -> str:
    """Get account information for the current user."""
    user = USER_DB.get(user_id)
    if not user:
        return f"No account found."
    return f"Account: balance=${user['balance']}, plan={user['plan']}"

# Inject user_id at runtime (not from LLM)
result = get_account_info.invoke({"query": "balance", "user_id": "alice"})
print(f"  Account info (alice): {result}")
print(f"  LLM schema (no user_id shown):")
schema = get_account_info.args_schema.model_json_schema()
visible_props = [k for k in schema.get("properties", {}).keys()]
print(f"    Visible args: {visible_props}  ← user_id is hidden from LLM!")

# ─────────────────────────────────────────────────────────────
# SECTION 5: Tool Metadata and Configuration
# ─────────────────────────────────────────────────────────────
print("\n📌 5. Tool Metadata")
print("─" * 40)

@tool(
    name="web_search",
    description="Search the internet for current information. Use for news, events, prices.",
    return_direct=False,   # agent uses result in further reasoning
)
def search_web(query: str, max_results: int = 3) -> str:
    """Mock web search."""
    return f"Mock results for '{query}': [Result 1] [Result 2] [Result 3]"

print(f"  Tool name        : {search_web.name}")
print(f"  Tool description : {search_web.description}")
print(f"  Tool return_direct: {search_web.return_direct}")
print(f"  Args schema      : {list(search_web.args.keys())}")

# ─────────────────────────────────────────────────────────────
# SECTION 6: Tool Chaining — output of one tool as input to another
# ─────────────────────────────────────────────────────────────
print("\n📌 6. Tools in a Chain (without an agent)")
print("─" * 40)

@tool
def get_city_population(city: str) -> str:
    """Get approximate population of a city."""
    populations = {"Mumbai": "20M", "Delhi": "32M", "Bangalore": "12M"}
    return populations.get(city, f"Population data not available for {city}")

@tool
def calculate_percentage(value: str, percentage: float) -> str:
    """Calculate percentage of a value like '20M'. Returns result."""
    num_map = {"M": 1_000_000, "K": 1_000, "B": 1_000_000_000}
    suffix  = value[-1] if value[-1] in num_map else ""
    num     = float(value.replace(suffix, "")) * num_map.get(suffix, 1)
    result  = num * (percentage / 100)
    return f"{percentage}% of {value} = {result:,.0f}"

# Manual tool chain: get city pop, then calculate a percentage
city     = "Mumbai"
pop      = get_city_population.invoke({"city": city})
result   = calculate_percentage.invoke({"value": pop, "percentage": 5.0})
print(f"  {city} population : {pop}")
print(f"  5% of population  : {result}")

# ─────────────────────────────────────────────────────────────
# ⚠️  COMMON MISTAKE 1: Raising exceptions inside tools
# Exceptions crash the agent. Always return error strings.
# BAD:  raise ValueError("City not found")
# GOOD: return "Error: City not found. Try: Mumbai, Delhi, Bangalore"

# ⚠️  COMMON MISTAKE 2: Tools with too many optional params
# The LLM gets confused by tools with many parameters.
# Keep each tool focused — one job, minimal required params.

# ⚠️  COMMON MISTAKE 3: Not hiding sensitive params with InjectedToolArg
# Never let the LLM see API keys, user IDs, or DB connections.
# ─────────────────────────────────────────────────────────────
print("\n⚠️  Advanced tools tips:")
print("   • Return error strings — never raise exceptions in tools")
print("   • Use InjectedToolArg for API keys, user IDs, sessions")
print("   • Use async tools for I/O-bound operations")
print("   • Keep tools focused: one job, minimal parameters")
