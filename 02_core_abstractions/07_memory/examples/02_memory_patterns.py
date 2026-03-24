"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 07 — Memory                                           ║
║  File:   02_memory_patterns.py                               ║
║  Level:  Advanced                                            ║
║  Goal:   Multi-session memory, summary compression,          ║
║          entity memory, and production memory patterns       ║
╚══════════════════════════════════════════════════════════════╝
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
llm0   = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

print("=" * 60)
print("07 — Memory Patterns")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# PATTERN 1: Multi-session memory with RunnableWithMessageHistory
# Each session_id maps to an independent conversation history.
# Different users / conversations stay isolated.
# ─────────────────────────────────────────────────────────────
print("\n📌 1. Multi-Session Memory")
print("─" * 40)

# In-memory store — in production, replace with Redis/DB
sessions: dict[str, InMemoryChatMessageHistory] = {}

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    """Returns (or creates) the message history for a session."""
    if session_id not in sessions:
        sessions[session_id] = InMemoryChatMessageHistory()
    return sessions[session_id]

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

chain = template | llm | parser

chat = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Separate sessions — completely isolated
session_a = {"configurable": {"session_id": "alice"}}
session_b = {"configurable": {"session_id": "bob"}}

print("  [Alice] Turn 1:", chat.invoke({"input": "I am Alice and I like cats."}, config=session_a))
print("  [Bob]   Turn 1:", chat.invoke({"input": "I am Bob and I like dogs."}, config=session_b))
print("  [Alice] Turn 2:", chat.invoke({"input": "What pet do I like?"}, config=session_a))
print("  [Bob]   Turn 2:", chat.invoke({"input": "What pet do I like?"}, config=session_b))

print(f"\n  Active sessions: {list(sessions.keys())}")
print(f"  Alice's history: {len(sessions['alice'].messages)} messages")
print(f"  Bob's history  : {len(sessions['bob'].messages)} messages")

# ─────────────────────────────────────────────────────────────
# PATTERN 2: Window Memory — keep only last N turns
# Prevents context window overflow for long conversations.
# ─────────────────────────────────────────────────────────────
print("\n📌 2. Window Memory — Last N Turns Only")
print("─" * 40)

class WindowedChat:
    """Chat that remembers only the last N turns."""

    def __init__(self, window_size: int = 3):
        self.window_size = window_size  # number of turns (each turn = 2 messages)
        self.history: list = []
        self.chain = (
            ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant."),
                MessagesPlaceholder("history"),
                ("human", "{input}"),
            ])
            | llm | parser
        )

    def chat(self, user_input: str) -> str:
        # Keep only the last N turns (window_size * 2 messages)
        windowed = self.history[-(self.window_size * 2):]
        response = self.chain.invoke({"input": user_input, "history": windowed})
        self.history.append(HumanMessage(content=user_input))
        self.history.append(AIMessage(content=response))
        return response

    @property
    def active_messages(self):
        return len(self.history[-(self.window_size * 2):])

windowed = WindowedChat(window_size=2)   # remembers last 2 turns only

messages = [
    "My name is Rahul.",
    "I work at a startup.",
    "I love Python.",
    "I have 5 years of experience.",
    "What's my name?",   # Can it remember from 4 turns ago?
]

for msg in messages:
    reply = windowed.chat(msg)
    print(f"  User  : {msg}")
    print(f"  Bot   : {reply[:80]}")
    print(f"  Active: {windowed.active_messages} messages in window\n")

# ─────────────────────────────────────────────────────────────
# PATTERN 3: Summary Memory — compress old history
# Instead of trimming history, summarize it to preserve context.
# ─────────────────────────────────────────────────────────────
print("\n📌 3. Summary Memory — Compress History")
print("─" * 40)

class SummaryMemoryChat:
    """Chat that summarizes history when it gets long."""

    SUMMARIZE_AFTER = 6  # messages before summarizing

    def __init__(self):
        self.summary = ""
        self.recent: list = []
        self.summarize_chain = (
            ChatPromptTemplate.from_template(
                "Summarize this conversation in 2-3 bullet points:\n{history}"
            ) | llm0 | parser
        )
        self.chat_chain = (
            ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant.\n\n"
                           "Conversation so far:\n{summary}"),
                MessagesPlaceholder("recent"),
                ("human", "{input}"),
            ])
            | llm | parser
        )

    def _maybe_summarize(self):
        """Compress history into summary if it's too long."""
        if len(self.recent) >= self.SUMMARIZE_AFTER:
            history_text = "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                for m in self.recent
            )
            self.summary = self.summarize_chain.invoke({"history": history_text})
            self.recent = []  # clear after summarizing
            print(f"  📝 [Summary created: {self.summary[:80]}...]")

    def chat(self, user_input: str) -> str:
        response = self.chat_chain.invoke({
            "input": user_input,
            "summary": self.summary or "No prior conversation.",
            "recent": self.recent,
        })
        self.recent.append(HumanMessage(content=user_input))
        self.recent.append(AIMessage(content=response))
        self._maybe_summarize()
        return response

summary_chat = SummaryMemoryChat()
conversation = [
    "My name is Priya.",
    "I am a data scientist.",
    "I use Python and SQL daily.",
    "My company is building an AI product.",
    "We are using LangChain for the backend.",  # triggers summarization
    "What stack did I mention I use?",           # relies on summary
]

for msg in conversation:
    reply = summary_chat.chat(msg)
    print(f"  User: {msg}")
    print(f"  Bot : {reply[:80]}\n")

# ─────────────────────────────────────────────────────────────
# PATTERN 4: Per-topic memory namespacing
# Different topics / modes get isolated histories.
# ─────────────────────────────────────────────────────────────
print("\n📌 4. Namespaced Memory (per topic/mode)")
print("─" * 40)

namespaced_sessions: dict[str, InMemoryChatMessageHistory] = {}

def get_namespaced_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in namespaced_sessions:
        namespaced_sessions[session_id] = InMemoryChatMessageHistory()
    return namespaced_sessions[session_id]

namespaced_chat = RunnableWithMessageHistory(
    chain, get_namespaced_history,
    input_messages_key="input",
    history_messages_key="history",
)

# user:rahul:python — Rahul's Python questions
# user:rahul:career — Rahul's career questions
# Each gets its own isolated history

config_python = {"configurable": {"session_id": "user:rahul:python"}}
config_career = {"configurable": {"session_id": "user:rahul:career"}}

namespaced_chat.invoke({"input": "Explain Python decorators."}, config=config_python)
namespaced_chat.invoke({"input": "I want to become a ML engineer."}, config=config_career)

r1 = namespaced_chat.invoke({"input": "What did I ask about?"}, config=config_python)
r2 = namespaced_chat.invoke({"input": "What did I ask about?"}, config=config_career)

print(f"  [python namespace]: {r1[:80]}")
print(f"  [career namespace]: {r2[:80]}")

# ─────────────────────────────────────────────────────────────
# ⚠️  COMMON MISTAKE 1: Reusing the same history across sessions
# Always key your history store by session_id.
# Sharing history between users is a privacy and logic bug.

# ⚠️  COMMON MISTAKE 2: Infinite history growth
# Without windowing or summarization, history grows unbounded.
# At ~16K tokens, costs spike and performance drops.
# Rule of thumb: start summarizing after 10-12 turns.

# ⚠️  COMMON MISTAKE 3: Storing AIMessage raw objects in DB
# Always serialize to dicts when persisting to a database.
# ─────────────────────────────────────────────────────────────
print("\n⚠️  Memory tips:")
print("   • Always key history by session_id (user isolation)")
print("   • Use window or summary memory for long chats")
print("   • In production, replace InMemoryStore with Redis/DB")
print("   • Namespace session_ids: 'user:<id>:<topic>'")
