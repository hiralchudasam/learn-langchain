"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 19 — LangServe                                        ║
║  File:   03_langserve_auth.py                                ║
║  Level:  Advanced                                            ║
║  Goal:   Add auth middleware, rate limiting, input           ║
║          validation, and health checks to LangServe APIs     ║
╚══════════════════════════════════════════════════════════════╝
Run: python 03_langserve_auth.py
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel
import time
from collections import defaultdict

app = FastAPI(title="Production LangServe API", version="1.0.0")

# CORS
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ─────────────────────────────────────────────────────────────
# AUTH: API Key validation
# ─────────────────────────────────────────────────────────────
API_KEYS = {"dev-key-123", "prod-key-abc"}
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# ─────────────────────────────────────────────────────────────
# RATE LIMITING: Simple in-memory limiter
# ─────────────────────────────────────────────────────────────
request_counts = defaultdict(list)
RATE_LIMIT     = 10   # requests per minute

def rate_limit_check(request: Request, api_key: str = Depends(verify_api_key)):
    now  = time.time()
    key  = api_key
    calls = [t for t in request_counts[key] if now - t < 60]
    if len(calls) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail=f"Rate limit: {RATE_LIMIT} req/min")
    request_counts[key] = calls + [now]
    return api_key

# ─────────────────────────────────────────────────────────────
# INPUT VALIDATION: Guard against bad inputs
# ─────────────────────────────────────────────────────────────
def validate_input(inputs: dict) -> dict:
    text = inputs.get("text", "") or inputs.get("topic", "")
    if not text or len(text.strip()) < 3:
        raise ValueError("Input too short (min 3 chars)")
    if len(text) > 1000:
        raise ValueError("Input too long (max 1000 chars)")
    blocked = ["ignore previous", "jailbreak", "system prompt"]
    if any(b in text.lower() for b in blocked):
        raise ValueError("Input contains blocked content")
    return inputs

# ─────────────────────────────────────────────────────────────
# CHAINS
# ─────────────────────────────────────────────────────────────
summarize_chain = (
    RunnableLambda(validate_input)
    | ChatPromptTemplate.from_template("Summarize in 2 sentences: {text}")
    | llm | StrOutputParser()
)

qa_chain = (
    RunnableLambda(validate_input)
    | ChatPromptTemplate.from_template("Answer briefly: {topic}")
    | llm | StrOutputParser()
)

# ─────────────────────────────────────────────────────────────
# REGISTER ROUTES with auth + rate limiting
# ─────────────────────────────────────────────────────────────
add_routes(
    app,
    summarize_chain,
    path="/summarize",
    dependencies=[Depends(rate_limit_check)],
)

add_routes(
    app,
    qa_chain,
    path="/qa",
    dependencies=[Depends(rate_limit_check)],
)

# ─────────────────────────────────────────────────────────────
# HEALTH CHECK ENDPOINT
# ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time(), "routes": ["/summarize", "/qa"]}

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc)})

if __name__ == "__main__":
    import uvicorn
    print("Starting authenticated LangServe API...")
    print("→ Add header: X-API-Key: dev-key-123")
    print("→ Health:     http://localhost:8001/health")
    print("→ QA:         http://localhost:8001/qa/playground")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
