"""
16 — LangGraph
Example 01: ReAct Agent with tools, conditional edges, checkpointing
"""

from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import json

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ─── 1. Define Tools ──────────────────────────────────────────────────────────

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city (mock)."""
    weather_db = {
        "mumbai": "Hot and humid, 34°C",
        "delhi": "Sunny, 28°C",
        "bangalore": "Pleasant, 24°C",
    }
    return weather_db.get(city.lower(), f"Weather data for {city} not available.")

tools = [add, multiply, get_weather]
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

# ─── 2. Define State ──────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ─── 3. Define Nodes ──────────────────────────────────────────────────────────

def call_llm(state: AgentState) -> AgentState:
    """Node: Call the LLM with current messages."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def call_tools(state: AgentState) -> AgentState:
    """Node: Execute all tool calls from the last AI message."""
    last_message = state["messages"][-1]
    tool_results = []

    for tool_call in last_message.tool_calls:
        tool_fn = tools_by_name[tool_call["name"]]
        result = tool_fn.invoke(tool_call["args"])
        tool_results.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
            )
        )
    return {"messages": tool_results}

# ─── 4. Define Routing Logic ──────────────────────────────────────────────────

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Route: If last AI message has tool calls → tools, else → end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

# ─── 5. Build the Graph ───────────────────────────────────────────────────────

graph = StateGraph(AgentState)

graph.add_node("llm", call_llm)
graph.add_node("tools", call_tools)

graph.add_edge(START, "llm")
graph.add_conditional_edges(
    "llm",
    should_continue,
    {"tools": "tools", "end": END},
)
graph.add_edge("tools", "llm")   # Loop back after tool execution

# ─── 6. Compile Without Checkpointing ─────────────────────────────────────────

app = graph.compile()

print("=" * 55)
print("1. Basic Agent (no memory)")
print("=" * 55)

result = app.invoke({
    "messages": [HumanMessage("What is 42 multiplied by 13, then add 7?")]
})
print("Final answer:", result["messages"][-1].content)

result2 = app.invoke({
    "messages": [HumanMessage("What's the weather in Mumbai and Bangalore?")]
})
print("\nWeather:", result2["messages"][-1].content)

# ─── 7. Compile WITH Checkpointing (persistent memory) ────────────────────────

print("\n" + "=" * 55)
print("2. Agent with Memory (checkpointing)")
print("=" * 55)

checkpointer = MemorySaver()
app_with_memory = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user_session_1"}}

turn1 = app_with_memory.invoke(
    {"messages": [HumanMessage("My name is Rahul. What is 100 * 5?")]},
    config=config,
)
print("Turn 1:", turn1["messages"][-1].content)

turn2 = app_with_memory.invoke(
    {"messages": [HumanMessage("What is my name? Also add 500 to our previous answer.")]},
    config=config,
)
print("Turn 2:", turn2["messages"][-1].content)

# ─── 8. Stream the graph ──────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("3. Streaming Graph Execution")
print("=" * 55)

for step in app.stream(
    {"messages": [HumanMessage("What is 15 + 27? Then multiply that by 2.")]},
    stream_mode="updates",
):
    node_name = list(step.keys())[0]
    messages = step[node_name]["messages"]
    last_msg = messages[-1]

    if isinstance(last_msg, AIMessage):
        if last_msg.tool_calls:
            calls = [f"{tc['name']}({tc['args']})" for tc in last_msg.tool_calls]
            print(f"[{node_name}] Calling tools: {', '.join(calls)}")
        else:
            print(f"[{node_name}] Final answer: {last_msg.content}")
    elif isinstance(last_msg, ToolMessage):
        print(f"[{node_name}] Tool result ({last_msg.name}): {last_msg.content}")

# ─── 9. Visualize the graph ───────────────────────────────────────────────────

print("\n" + "=" * 55)
print("4. Graph Structure")
print("=" * 55)
print(app.get_graph().draw_ascii())
