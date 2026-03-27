"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 15 — Agents                                           ║
║  File:   01_agents_basics.py                                 ║
║  Level:  Basic → Intermediate                                ║
║  Goal:   Build ReAct agents with tools, understand the       ║
║          agent loop, and use create_react_agent              ║
╚══════════════════════════════════════════════════════════════╝
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("=" * 60)
print("15 — Agents Basics")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# SECTION 1: Understanding the Agent Loop
# Without agents: chain follows a fixed path A → B → C
# With agents:    LLM decides what to do next at each step
#
# The ReAct loop:
#   Reason → Act (call tool) → Observe (see result) → Reason → ...
#   Stops when the LLM produces a final answer (no more tool calls)
# ─────────────────────────────────────────────────────────────
print("\n📌 1. Manual ReAct Loop — See How Agents Work")
print("─" * 40)

# Define simple tools
@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    print(f"     [tool: add({a}, {b})]")
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    print(f"     [tool: multiply({a}, {b})]")
    return a * b

@tool
def square_root(n: float) -> float:
    """Return the square root of a number."""
    import math
    print(f"     [tool: square_root({n})]")
    return round(math.sqrt(n), 4)

@tool
def get_country_capital(country: str) -> str:
    """Get the capital city of a country."""
    print(f"     [tool: get_country_capital({country})]")
    capitals = {
        "india": "New Delhi", "france": "Paris", "japan": "Tokyo",
        "brazil": "Brasília", "usa": "Washington D.C.", "germany": "Berlin",
    }
    return capitals.get(country.lower(), f"Unknown capital for {country}")

tools = [add, multiply, square_root, get_country_capital]
tools_map = {t.name: t for t in tools}

# Bind tools to LLM — tells the model what tools are available
llm_with_tools = llm.bind_tools(tools)

def run_agent_manually(question: str) -> str:
    """
    Manual implementation of the ReAct agent loop.
    This is exactly what create_react_agent does internally.
    """
    messages = [HumanMessage(content=question)]
    step = 0

    while True:
        step += 1
        print(f"   Step {step}: Calling LLM...")
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # If no tool calls → LLM is done, return the answer
        if not response.tool_calls:
            print(f"   → Final answer reached after {step} steps")
            return response.content

        # Execute each tool call
        for tc in response.tool_calls:
            tool_fn = tools_map[tc["name"]]
            result  = tool_fn.invoke(tc["args"])
            print(f"   Tool result: {result}")
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"])
            )

        if step > 10:  # safety guard
            return "Max steps exceeded"

print("Question: What is (15 + 7) * 3?")
answer = run_agent_manually("What is (15 + 7) * 3?")
print(f"Answer: {answer}\n")

print("Question: What is the capital of Japan and square root of 144?")
answer = run_agent_manually("What is the capital of Japan and the square root of 144?")
print(f"Answer: {answer}")

# ─────────────────────────────────────────────────────────────
# SECTION 2: create_react_agent — the easy way
# LangGraph's prebuilt ReAct agent — no manual loop needed.
# Handles the loop, history, tool calling, and stopping automatically.
# ─────────────────────────────────────────────────────────────
print("\n📌 2. create_react_agent (LangGraph prebuilt)")
print("─" * 40)

agent = create_react_agent(llm, tools)

# .invoke() returns the full state including all messages
result = agent.invoke({"messages": [HumanMessage("What is 100 * 7 + 50?")]})
final_message = result["messages"][-1]
print(f"  Answer: {final_message.content}")

# Show the agent's reasoning trace
print(f"\n  Full trace ({len(result['messages'])} messages):")
for msg in result["messages"]:
    msg_type = type(msg).__name__
    if isinstance(msg, HumanMessage):
        print(f"    👤 Human    : {msg.content}")
    elif isinstance(msg, AIMessage):
        if msg.tool_calls:
            calls = [f"{tc['name']}({tc['args']})" for tc in msg.tool_calls]
            print(f"    🤖 AI calls : {', '.join(calls)}")
        else:
            print(f"    🤖 AI answer: {msg.content}")
    elif isinstance(msg, ToolMessage):
        print(f"    🔧 Tool ({msg.name}): {msg.content}")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Streaming the agent's steps
# See each step as it happens instead of waiting for the final answer.
# ─────────────────────────────────────────────────────────────
print("\n📌 3. Streaming Agent Steps")
print("─" * 40)

print("  Question: What is the capital of France, and multiply it's population by 2?")
print("  (Streaming mode):\n")

for step in agent.stream(
    {"messages": [HumanMessage("What is the capital of France?")]},
    stream_mode="updates",
):
    node = list(step.keys())[0]
    msgs = step[node]["messages"]
    for msg in msgs:
        if isinstance(msg, AIMessage) and msg.content:
            print(f"    [{node}] {msg.content[:80]}")
        elif isinstance(msg, ToolMessage):
            print(f"    [{node}] Tool({msg.name}) → {msg.content}")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Agent with system prompt
# Guide the agent's behavior with a custom system message.
# ─────────────────────────────────────────────────────────────
print("\n📌 4. Agent with Custom System Prompt")
print("─" * 40)

from langchain_core.messages import SystemMessage

SYSTEM = """You are a helpful math and geography assistant.
- Always show your work step by step
- Round decimal results to 2 decimal places
- Be concise in your final answer"""

agent_with_system = create_react_agent(llm, tools, state_modifier=SYSTEM)

result = agent_with_system.invoke({
    "messages": [HumanMessage("What is the square root of the population steps: first find 100*100, then take sqrt")]
})
print(f"  {result['messages'][-1].content}")

# ─────────────────────────────────────────────────────────────
# ⚠️  COMMON MISTAKE 1: Expecting agents to always call tools
# If the LLM can answer from memory, it WON'T call a tool.
# Agents only call tools when they decide it's necessary.

# ⚠️  COMMON MISTAKE 2: No max_iterations guard
# A buggy tool or bad prompt can cause infinite loops.
# Always set a max_iterations limit in production.

# ⚠️  COMMON MISTAKE 3: Weak tool descriptions
# The LLM picks tools based on their docstring.
# A vague docstring = wrong tool chosen or tool not used.
# Write tool descriptions as if explaining to a person.
# ─────────────────────────────────────────────────────────────
print("\n⚠️  Agent tips:")
print("   • Agents decide whether to use tools — don't assume they always will")
print("   • Always set a max step/iteration limit in production")
print("   • Write clear, specific tool docstrings — the LLM reads them")
print("   • Use stream_mode='updates' to monitor agent reasoning")
