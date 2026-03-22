# 01 — Introduction to LangChain

## What is LangChain?

LangChain is an open-source framework for building applications powered by Large Language Models (LLMs). It provides a standard interface for chaining together different components — models, prompts, memory, tools — so you can build complex AI applications with minimal boilerplate.

Think of LangChain as the **glue layer** between your application and the LLM.

```
Your App → LangChain → LLM (OpenAI / Claude / Gemini / local)
```

---

## Why LangChain?

Without LangChain, building an LLM app means:
- Writing raw API calls to OpenAI/Anthropic
- Manually formatting prompts
- Handling conversation history yourself
- Connecting to databases and tools from scratch

With LangChain:
- Composable building blocks (prompts, models, parsers, tools)
- Built-in memory and conversation management
- 100+ integrations (vector stores, document loaders, tools)
- First-class support for agents and multi-step reasoning

---

## LangChain vs Alternatives

| Feature | LangChain | LlamaIndex | Haystack |
|---------|-----------|------------|---------|
| General purpose | ✅ | ⚠️ RAG-focused | ✅ |
| RAG support | ✅ | ✅ Excellent | ✅ |
| Agent support | ✅ Excellent | ✅ | ⚠️ Limited |
| Integrations | ✅ 100+ | ✅ 50+ | ✅ 50+ |
| LCEL / pipelines | ✅ | ❌ | ❌ |
| Production-ready | ✅ LangServe | ⚠️ | ⚠️ |
| Community | Very large | Large | Medium |

---

## LangChain Ecosystem

```
┌─────────────────────────────────────────────────────────────┐
│                     LangChain Ecosystem                     │
│                                                             │
│  langchain-core      ← Interfaces, base classes, LCEL       │
│  langchain           ← Chains, agents, memory               │
│  langchain-community ← 3rd party integrations               │
│  langchain-openai    ← OpenAI-specific package              │
│  langchain-anthropic ← Anthropic-specific package           │
│  langgraph           ← Stateful multi-agent graphs          │
│  langserve           ← Deploy chains as REST APIs           │
│  langsmith           ← Observability & evaluation           │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
# Minimal install
pip install langchain langchain-openai python-dotenv

# Full install (everything in this repo)
pip install -r requirements.txt
```

---

## Project Structure Pattern

Every topic in this repo follows this structure:

```
topic/
├── README.md         ← You are here — theory & concepts
├── examples/
│   ├── 01_basic.py
│   ├── 02_intermediate.py
│   └── 03_advanced.py
└── assignment/
    ├── README.md     ← Your task
    └── solution.py   ← Reference solution
```

---

## Key Concepts Glossary

| Term | Meaning |
|------|---------|
| **LLM** | Large Language Model (GPT-4, Claude, Gemini…) |
| **Chain** | A sequence of calls to models/tools/parsers |
| **LCEL** | LangChain Expression Language — pipe-based composition |
| **Runnable** | Any object that can be `.invoke()`-d or `.stream()`-d |
| **Agent** | LLM that decides which tools to call and when |
| **RAG** | Retrieval-Augmented Generation — search then generate |
| **Embedding** | Numerical vector representing meaning of text |
| **Vector Store** | Database that stores and searches embeddings |

---

## Next Topic

→ [02 — Language Models](../02_language_models/README.md)
