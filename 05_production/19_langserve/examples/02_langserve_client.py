"""
19 — LangServe
Example 02: Client — consuming LangServe APIs with RemoteRunnable
Run the server first: python 01_langserve_server.py
"""

import asyncio
from langserve import RemoteRunnable

BASE_URL = "http://localhost:8000"

# ─── Connect to remote chains ─────────────────────────────────────────────────

qa         = RemoteRunnable(f"{BASE_URL}/qa")
translator = RemoteRunnable(f"{BASE_URL}/translate")
code_chain = RemoteRunnable(f"{BASE_URL}/explain-code")
summarizer = RemoteRunnable(f"{BASE_URL}/summarize")

# ─── 1. invoke() ──────────────────────────────────────────────────────────────

print("=" * 50)
print("1. invoke()")
print("=" * 50)

answer = qa.invoke({"question": "What is LangChain?"})
print(f"QA: {answer}")

translation = translator.invoke({"text": "Hello, how are you?", "language": "Hindi"})
print(f"\nTranslation: {translation}")

# ─── 2. stream() ──────────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("2. stream()")
print("=" * 50)

print("Streaming code explanation: ", end="", flush=True)
for chunk in code_chain.stream({
    "language": "Python",
    "code": "result = [x**2 for x in range(10) if x % 2 == 0]",
}):
    print(chunk, end="", flush=True)
print()

# ─── 3. batch() ───────────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("3. batch() — parallel calls")
print("=" * 50)

questions = [
    {"question": "What is RAG?"},
    {"question": "What is LangGraph?"},
    {"question": "What is LangSmith?"},
]

answers = qa.batch(questions)
for q, a in zip(questions, answers):
    print(f"\nQ: {q['question']}")
    print(f"A: {a[:100]}...")

# ─── 4. Async ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("4. Async invoke + stream")
print("=" * 50)

async def run_async():
    # Async invoke
    result = await qa.ainvoke({"question": "What are embeddings?"})
    print(f"Async answer: {result[:100]}...")

    # Async stream
    print("\nAsync streaming: ", end="", flush=True)
    async for chunk in summarizer.astream({
        "text": "LangChain is an open-source framework for building LLM-powered apps. "
                "It supports chains, agents, memory, and retrieval. "
                "The latest version uses LCEL for composable pipelines.",
        "sentences": "2",
    }):
        print(chunk, end="", flush=True)
    print()

asyncio.run(run_async())

print("\n✅ All client calls completed!")
print("→ Check server logs and LangSmith for traces")
