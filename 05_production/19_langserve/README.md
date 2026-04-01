# 19 — LangServe

## What is LangServe?

LangServe lets you deploy any LCEL chain as a REST API with one function call. It builds on FastAPI and automatically generates:
- `POST /chain/invoke` — single call
- `POST /chain/batch` — multiple calls
- `POST /chain/stream` — streaming
- `GET /chain/playground` — interactive UI for testing

## Setup

```bash
pip install "langserve[all]"
```

## Basic Server

```python
# server.py
from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="LangChain API")

chain = (
    ChatPromptTemplate.from_template("Tell me a joke about {topic}")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

add_routes(app, chain, path="/joke")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

```bash
python server.py
# API live at http://localhost:8000
# Playground at http://localhost:8000/joke/playground
```

## Client

```python
from langserve import RemoteRunnable

chain = RemoteRunnable("http://localhost:8000/joke")
result = chain.invoke({"topic": "Python"})

# Supports all Runnable methods
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="")
```

## Adding Auth

```python
from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

def verify_key(key: str = Depends(api_key_header)):
    if key != "my-secret-key":
        raise HTTPException(status_code=403)

add_routes(app, chain, path="/joke", dependencies=[Depends(verify_key)])
```

## Next Topic
→ [20 — Evaluation](../20_evaluation/README.md)
