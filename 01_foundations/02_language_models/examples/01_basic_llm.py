"""
02 — Language Models
Example 01: Basic LLM and ChatModel usage
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

# ─── 1. Basic ChatModel ───────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

response = llm.invoke([HumanMessage(content="What is LangChain in one sentence?")])
print("Basic response:", response.content)
print("Token usage:", response.usage_metadata)

# ─── 2. With System Message ───────────────────────────────────────────────────

messages = [
    SystemMessage(content="You are a helpful Python tutor. Keep answers concise."),
    HumanMessage(content="What is a list comprehension?"),
]

response = llm.invoke(messages)
print("\nWith system message:", response.content)

# ─── 3. Conversation (multi-turn) ─────────────────────────────────────────────

history = [
    HumanMessage(content="My name is Rahul."),
    AIMessage(content="Nice to meet you, Rahul! How can I help you today?"),
    HumanMessage(content="What is my name?"),
]

response = llm.invoke(history)
print("\nConversation:", response.content)

# ─── 4. Streaming ─────────────────────────────────────────────────────────────

print("\nStreaming response:")
for chunk in llm.stream([HumanMessage(content="Count from 1 to 5 slowly.")]):
    print(chunk.content, end="", flush=True)
print()

# ─── 5. Model Parameters ──────────────────────────────────────────────────────

creative_llm = ChatOpenAI(model="gpt-4o-mini", temperature=1.5, max_tokens=50)
factual_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

print("\nCreative (temp=1.5):", creative_llm.invoke([HumanMessage("Tell me a creative word.")]).content)
print("Factual  (temp=0.0):", factual_llm.invoke([HumanMessage("What is 2+2?")]).content)

# ─── 6. Batch Calls ───────────────────────────────────────────────────────────

questions = [
    [HumanMessage(content="Capital of France?")],
    [HumanMessage(content="Capital of Japan?")],
    [HumanMessage(content="Capital of India?")],
]

responses = llm.batch(questions)
for q, r in zip(questions, responses):
    print(f"Q: {q[0].content} → A: {r.content}")
