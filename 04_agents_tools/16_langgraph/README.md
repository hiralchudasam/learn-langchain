# 16 — LangGraph

## What is LangGraph?

LangGraph is a library for building **stateful, multi-step, multi-agent** applications using a graph structure. While LCEL handles linear chains, LangGraph handles **cycles, conditionals, and loops** — things that agents need.

```
LCEL:     A → B → C → D       (linear, no cycles)
LangGraph: A → B → C → A      (cycles allowed — real agents loop!)
```

---

## Core Concepts

### State
A shared dictionary (TypedDict) that every node can read and write. The graph passes state from node to node.

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # auto-appends messages
    next_step: str
```

### Nodes
Python functions that take state and return updated state.

```python
def call_llm(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

### Edges
Connect nodes. Can be:
- **Normal edge**: always goes from A to B
- **Conditional edge**: chooses next node based on state

```python
graph.add_edge("node_a", "node_b")  # always goes to node_b

graph.add_conditional_edges(
    "node_a",
    should_continue,               # function that returns node name
    {"continue": "node_b", "end": END}
)
```

### START and END
Special nodes. `START` is the entry point. `END` terminates the graph.

---

## Basic Graph Structure

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(AgentState)

graph.add_node("llm_node", call_llm)
graph.add_node("tool_node", call_tools)

graph.add_edge(START, "llm_node")
graph.add_conditional_edges("llm_node", route, {"tools": "tool_node", "end": END})
graph.add_edge("tool_node", "llm_node")  # loop back!

app = graph.compile()
result = app.invoke({"messages": [HumanMessage("What is 42 * 13?")]})
```

---

## ReAct Agent Pattern

The classic agent loop:
```
START → LLM → (has tool calls?) → YES → Tools → LLM → ...
                                 → NO  → END
```

---

## Human-in-the-Loop

```python
app = graph.compile(interrupt_before=["tool_node"])  # pause before tools

# Run until interrupt
state = app.invoke(initial_state)
# ... show user what tools will be called ...
# Resume
state = app.invoke(None, config)
```

---

## Persistence & Checkpointing

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# Same thread_id = same conversation
config = {"configurable": {"thread_id": "user_123"}}
app.invoke({"messages": [HumanMessage("Hello")]}, config=config)
app.invoke({"messages": [HumanMessage("What did I just say?")]}, config=config)
```

---

## LangGraph vs AgentExecutor

| Feature | AgentExecutor | LangGraph |
|---------|--------------|-----------|
| Cycles | ✅ | ✅ |
| Custom logic | ⚠️ Limited | ✅ Full control |
| Human-in-loop | ⚠️ | ✅ First-class |
| Multi-agent | ❌ | ✅ |
| Checkpointing | ❌ | ✅ |
| Streaming | ✅ | ✅ |
| Recommended | Legacy | ✅ Modern |

---

## Next Topic

→ [17 — Callbacks & Tracing](../17_callbacks_tracing/README.md)
