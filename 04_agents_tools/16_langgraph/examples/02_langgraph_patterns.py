"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 16 — LangGraph                                        ║
║  File:   02_langgraph_patterns.py                            ║
║  Level:  Advanced                                            ║
║  Goal:   Conditional routing, human-in-the-loop,             ║
║          multi-agent subgraphs, and state management         ║
╚══════════════════════════════════════════════════════════════╝
"""

from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("=" * 60)
print("16 — LangGraph Advanced Patterns")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# PATTERN 1: Conditional Routing
# Route to different nodes based on state content.
# The routing function returns the NAME of the next node.
# ─────────────────────────────────────────────────────────────
print("\n📌 1. Conditional Routing — Classify → Route")
print("─" * 40)

class RouterState(TypedDict):
    messages:  Annotated[list[BaseMessage], add_messages]
    category:  str   # "math", "geography", "general"
    answer:    str

def classify_node(state: RouterState) -> RouterState:
    """Classify the type of question."""
    question = state["messages"][-1].content
    prompt   = f"Classify this question as exactly one of: math, geography, general\nQuestion: {question}\nCategory:"
    response = llm.invoke([HumanMessage(content=prompt)])
    category = response.content.strip().lower()
    if category not in ["math", "geography", "general"]:
        category = "general"
    return {"category": category}

def math_node(state: RouterState) -> RouterState:
    question = state["messages"][-1].content
    response = llm.invoke([
        HumanMessage(content=f"You are a math expert. Solve: {question}")
    ])
    return {"messages": [response], "answer": response.content}

def geography_node(state: RouterState) -> RouterState:
    question = state["messages"][-1].content
    response = llm.invoke([
        HumanMessage(content=f"You are a geography expert. Answer: {question}")
    ])
    return {"messages": [response], "answer": response.content}

def general_node(state: RouterState) -> RouterState:
    question = state["messages"][-1].content
    response = llm.invoke([HumanMessage(content=question)])
    return {"messages": [response], "answer": response.content}

def route_by_category(state: RouterState) -> Literal["math", "geography", "general"]:
    """Routing function — returns the name of the next node."""
    return state.get("category", "general")

# Build the routing graph
router_graph = StateGraph(RouterState)
router_graph.add_node("classify",  classify_node)
router_graph.add_node("math",      math_node)
router_graph.add_node("geography", geography_node)
router_graph.add_node("general",   general_node)

router_graph.add_edge(START, "classify")
router_graph.add_conditional_edges(
    "classify",
    route_by_category,
    {"math": "math", "geography": "geography", "general": "general"},
)
router_graph.add_edge("math",      END)
router_graph.add_edge("geography", END)
router_graph.add_edge("general",   END)

router_app = router_graph.compile()

test_questions = [
    "What is 15 * 24?",
    "What is the capital of Brazil?",
    "What is machine learning?",
]

for question in test_questions:
    result = router_app.invoke({"messages": [HumanMessage(question)]})
    print(f"  [{result['category']:<10}] Q: {question}")
    print(f"             A: {result['answer'][:80]}\n")

# ─────────────────────────────────────────────────────────────
# PATTERN 2: Human-in-the-Loop
# Pause the graph before a sensitive step.
# A human reviews and approves before continuing.
# ─────────────────────────────────────────────────────────────
print("\n📌 2. Human-in-the-Loop")
print("─" * 40)

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email (mock). REQUIRES human approval before sending."""
    return f"Email sent to {to}: '{subject}'"

@tool
def draft_email(topic: str) -> str:
    """Draft an email on a given topic."""
    return f"Draft: Subject='Re: {topic}', Body='Hello, regarding {topic}...'"

email_tools = [send_email, draft_email]
email_llm   = llm.bind_tools(email_tools)

class EmailState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def email_agent(state: EmailState) -> EmailState:
    return {"messages": [email_llm.invoke(state["messages"])]}

def should_continue_email(state: EmailState) -> Literal["tools", "end"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"

email_graph = StateGraph(EmailState)
email_graph.add_node("agent", email_agent)
email_graph.add_node("tools", ToolNode(email_tools))
email_graph.add_edge(START, "agent")
email_graph.add_conditional_edges("agent", should_continue_email,
                                   {"tools": "tools", "end": END})
email_graph.add_edge("tools", "agent")

# ← Pause BEFORE executing tools (human reviews tool calls)
checkpointer = MemorySaver()
email_app = email_graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["tools"],
)

config = {"configurable": {"thread_id": "email_session_1"}}

# Step 1: Start — agent prepares tool calls
state = email_app.invoke(
    {"messages": [HumanMessage("Draft and send an email about the LangChain meeting to boss@company.com")]},
    config=config,
)

# Inspect what the agent wants to do
last_msg = state["messages"][-1]
if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
    print(f"  ⏸️  PAUSED — Agent wants to call:")
    for tc in last_msg.tool_calls:
        print(f"     Tool: {tc['name']}({tc['args']})")
    print(f"  → In production: show this to user for approval")
    print(f"  → Resuming automatically for demo...")

    # Step 2: Resume (human approved)
    final = email_app.invoke(None, config=config)
    print(f"  ✅ Final answer: {final['messages'][-1].content}")

# ─────────────────────────────────────────────────────────────
# PATTERN 3: Persistent Memory Across Sessions
# Same thread_id = same conversation history, even across restarts.
# ─────────────────────────────────────────────────────────────
print("\n📌 3. Persistent Memory (same thread = same conversation)")
print("─" * 40)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState) -> ChatState:
    return {"messages": [llm.invoke(state["messages"])]}

chat_graph = StateGraph(ChatState)
chat_graph.add_node("chat", chat_node)
chat_graph.add_edge(START, "chat")
chat_graph.add_edge("chat", END)

memory = MemorySaver()
chat_app = chat_graph.compile(checkpointer=memory)

thread = {"configurable": {"thread_id": "user_rahul"}}

r1 = chat_app.invoke({"messages": [HumanMessage("My name is Rahul.")]}, config=thread)
print(f"  Turn 1: {r1['messages'][-1].content[:80]}")

r2 = chat_app.invoke({"messages": [HumanMessage("What is my name?")]}, config=thread)
print(f"  Turn 2: {r2['messages'][-1].content[:80]}")
print(f"  ← Name remembered across turns using MemorySaver + thread_id!")

# Check the stored state
snapshot = chat_app.get_state(thread)
print(f"\n  State has {len(snapshot.values['messages'])} messages in memory")

# ─────────────────────────────────────────────────────────────
# ⚠️  COMMON MISTAKE 1: Forgetting add_messages annotation
# Without Annotated[list, add_messages], each node REPLACES the list.
# With it, each node APPENDS to the list.
# BAD:  messages: list[BaseMessage]
# GOOD: messages: Annotated[list[BaseMessage], add_messages]

# ⚠️  COMMON MISTAKE 2: Not compiling with checkpointer for memory
# graph.compile() → no memory
# graph.compile(checkpointer=MemorySaver()) → persistent per thread_id

# ⚠️  COMMON MISTAKE 3: Same thread_id for different users
# thread_id is the session key. Different users MUST have different IDs.
# ─────────────────────────────────────────────────────────────
print("\n⚠️  LangGraph tips:")
print("   • Always use Annotated[list, add_messages] for message state")
print("   • Use MemorySaver + thread_id for per-session memory")
print("   • Use interrupt_before for human-in-the-loop approval flows")
print("   • Routing functions must return the exact node name string")
