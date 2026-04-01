"""
19 — LangServe
Example 01: Deploy chains as REST APIs with FastAPI + LangServe
Run: python 01_langserve_server.py
Then visit: http://localhost:8000/docs  or  http://localhost:8000/chat/playground
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from typing import List

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LangChain API",
    description="LangChain chains deployed as REST APIs using LangServe",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ─── Chain 1: Simple Q&A ──────────────────────────────────────────────────────

qa_chain = (
    ChatPromptTemplate.from_template("Answer this question concisely: {question}")
    | llm
    | StrOutputParser()
)

add_routes(
    app,
    qa_chain,
    path="/qa",
    # This creates:
    # POST /qa/invoke    → single call
    # POST /qa/batch     → multiple calls
    # POST /qa/stream    → streaming
    # GET  /qa/playground → interactive UI
)

# ─── Chain 2: Translator ──────────────────────────────────────────────────────

translate_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a professional translator. Translate the given text to {language}. Only return the translation, nothing else."),
        ("human", "{text}"),
    ])
    | llm
    | StrOutputParser()
)

add_routes(app, translate_chain, path="/translate")

# ─── Chain 3: Code Explainer ──────────────────────────────────────────────────

code_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a senior developer. Explain code clearly and concisely."),
        ("human", "Explain this {language} code:\n\n```{language}\n{code}\n```"),
    ])
    | llm
    | StrOutputParser()
)

add_routes(app, code_chain, path="/explain-code")

# ─── Chain 4: Summarizer ──────────────────────────────────────────────────────

summarize_chain = (
    ChatPromptTemplate.from_template(
        "Summarize the following text in {sentences} sentences:\n\n{text}"
    )
    | llm
    | StrOutputParser()
)

add_routes(app, summarize_chain, path="/summarize")

# ─── Custom Endpoint (not LangServe) ─────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    routes: List[str]

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        routes=["/qa", "/translate", "/explain-code", "/summarize"],
    )

# ─── Run Server ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("Starting LangServe API...")
    print("→ Docs:       http://localhost:8000/docs")
    print("→ QA:         http://localhost:8000/qa/playground")
    print("→ Translate:  http://localhost:8000/translate/playground")
    print("→ Code:       http://localhost:8000/explain-code/playground")
    print("→ Summarize:  http://localhost:8000/summarize/playground")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
