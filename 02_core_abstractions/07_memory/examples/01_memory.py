"""
07 — Memory
Example 01: Manual history management, RunnableWithMessageHistory, ConversationSummaryMemory
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ─── 1. Manual History Management ─────────────────────────────────────────────

print("=" * 50)
print("1. Manual History")
print("=" * 50)

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Remember details about the user."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

chain = template | llm | StrOutputParser()
history = []

def chat_manual(user_input: str) -> str:
    response = chain.invoke({"input": user_input, "history": history})
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response))
    return response

print("User: My name is Rahul and I am learning LangChain.")
print("Bot:", chat_manual("My name is Rahul and I am learning LangChain."))
print("\nUser: What is my name?")
print("Bot:", chat_manual("What is my name?"))
print("\nUser: What am I learning?")
print("Bot:", chat_manual("What am I learning?"))

# ─── 2. RunnableWithMessageHistory ────────────────────────────────────────────

print("\n" + "=" * 50)
print("2. RunnableWithMessageHistory")
print("=" * 50)

# In-memory store keyed by session_id
store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

template2 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful tutor."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

chain2 = template2 | llm | StrOutputParser()

with_history = RunnableWithMessageHistory(
    chain2,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

config_a = {"configurable": {"session_id": "user_a"}}
config_b = {"configurable": {"session_id": "user_b"}}

print("Session A — Turn 1:", with_history.invoke({"input": "I am learning Python."}, config=config_a))
print("Session B — Turn 1:", with_history.invoke({"input": "I am learning Java."}, config=config_b))
print("Session A — Turn 2:", with_history.invoke({"input": "What language am I learning?"}, config=config_a))
print("Session B — Turn 2:", with_history.invoke({"input": "What language am I learning?"}, config=config_b))

# ─── 3. ConversationSummaryMemory pattern ─────────────────────────────────────

print("\n" + "=" * 50)
print("3. Summary Memory Pattern (manual)")
print("=" * 50)

summary = ""
conversation_history = []

def summarize_history(history: list) -> str:
    if not history:
        return ""
    summary_chain = (
        ChatPromptTemplate.from_template(
            "Summarize this conversation in 2-3 sentences:\n{history}"
        )
        | llm | StrOutputParser()
    )
    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in history
    )
    return summary_chain.invoke({"history": history_text})

summary_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant.\n\nConversation summary so far:\n{summary}"),
    MessagesPlaceholder("recent"),
    ("human", "{input}"),
])

summary_chain = summary_template | llm | StrOutputParser()

def chat_with_summary(user_input: str) -> str:
    global summary, conversation_history
    response = summary_chain.invoke({
        "input": user_input,
        "summary": summary,
        "recent": conversation_history[-4:],  # keep last 2 turns
    })
    conversation_history.append(HumanMessage(content=user_input))
    conversation_history.append(AIMessage(content=response))
    # Re-summarize after every 6 messages
    if len(conversation_history) >= 6:
        summary = summarize_history(conversation_history)
        conversation_history = []
        print(f"  [Summary updated: {summary[:60]}...]")
    return response

print("Bot:", chat_with_summary("I am Rahul, a developer from India."))
print("Bot:", chat_with_summary("I mostly work with Python and FastAPI."))
print("Bot:", chat_with_summary("Who am I?"))
