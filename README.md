<div align="center">

```
██╗      █████╗ ███╗   ██╗ ██████╗  ██████╗██╗  ██╗ █████╗ ██╗███╗   ██╗
██║     ██╔══██╗████╗  ██║██╔════╝ ██╔════╝██║  ██║██╔══██╗██║████╗  ██║
██║     ███████║██╔██╗ ██║██║  ███╗██║     ███████║███████║██║██╔██╗ ██║
██║     ██╔══██║██║╚██╗██║██║   ██║██║     ██╔══██║██╔══██║██║██║╚██╗██║
███████╗██║  ██║██║ ╚████║╚██████╔╝╚██████╗██║  ██║██║  ██║██║██║ ╚████║
╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝
```


### 🦜🔗 Build context-aware, reasoning applications with LangChain

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3144?style=flat-square&logo=chainlink&logoColor=white)](https://python.langchain.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Stars](https://img.shields.io/github/stars/YOUR_USERNAME/langchain-learning?style=flat-square&color=F59E0B)](https://github.com/YOUR_USERNAME/langchain-learning)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-8B5CF6?style=flat-square)](CONTRIBUTING.md)

</div>

---

## 🌐 What is LangChain?

**LangChain** is the most popular open-source framework for building production-ready applications powered by Large Language Models. It gives you composable building blocks to go from a simple LLM call to a full autonomous agent — without reinventing the wheel.

```
Your App ──────► LangChain ──────► OpenAI / Claude / Gemini / Local LLMs
                     │
              ┌──────┴───────┐
              │   Prompts    │  ← Structured, reusable prompt templates
              │   Chains     │  ← Composable sequences using LCEL
              │   Memory     │  ← Conversation history & summarization
              │   Agents     │  ← LLM-driven tool selection & execution
              │   RAG        │  ← Retrieval over your private data
              │   LangGraph  │  ← Stateful multi-agent orchestration
              └──────────────┘
```

> **Why LangChain?** Without it, you'd write raw API calls, manually manage conversation history, wire up vector databases from scratch, and build tool-calling logic by hand. LangChain handles all of that — so you focus on what your app does, not how it talks to the LLM.

---

## 📦 The LangChain Ecosystem

| Package | Purpose | Install |
|---------|---------|---------|
| `langchain-core` | Base interfaces, LCEL, Runnable | Auto-installed |
| `langchain` | Chains, agents, memory | `pip install langchain` |
| `langchain-community` | 100+ third-party integrations | `pip install langchain-community` |
| `langchain-openai` | OpenAI LLMs & embeddings | `pip install langchain-openai` |
| `langchain-anthropic` | Claude LLMs | `pip install langchain-anthropic` |
| `langgraph` | Stateful agent graphs | `pip install langgraph` |
| `langserve` | Deploy chains as REST APIs | `pip install langserve` |
| `langsmith` | Observability & evaluation | `pip install langsmith` |

---

## ⚡ Quickstart

### 1. Install

```bash
pip install langchain langchain-openai python-dotenv
```

### 2. Set your API key

```bash
# .env
OPENAI_API_KEY=your_key_here
```

### 3. Your first chain

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

chain = (
    ChatPromptTemplate.from_template("Explain {topic} in simple terms.")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

print(chain.invoke({"topic": "RAG"}))
```

That's it — a prompt template, a model, and a parser, connected with `|`. This is **LCEL** (LangChain Expression Language).

---

## 🗺️ Curriculum

This repo is a complete, structured learning path from beginner to production. Every topic has **theory**, **working code**, and **an assignment**.

<details>
<summary><b>📘 Phase 1 — Foundations</b> (4 topics)</summary>

| # | Topic | What You'll Learn |
|---|-------|------------------|
| 01 | [Introduction](./01_foundations/01_introduction/README.md) | What LangChain is, ecosystem overview, installation |
| 02 | [Language Models](./01_foundations/02_language_models/README.md) | LLM vs ChatModel, providers, streaming, batching |
| 03 | [Prompt Templates](./01_foundations/03_prompt_templates/README.md) | PromptTemplate, ChatPromptTemplate, FewShot, Partial |
| 04 | [Output Parsers](./01_foundations/04_output_parsers/README.md) | StrOutputParser, PydanticOutputParser, JsonOutputParser |

</details>

<details>
<summary><b>📗 Phase 2 — Core Abstractions</b> (4 topics)</summary>

| # | Topic | What You'll Learn |
|---|-------|------------------|
| 05 | [LCEL](./02_core_abstractions/05_lcel/README.md) | Pipe operator, Parallel, Passthrough, Lambda, Branch |
| 06 | [Chains](./02_core_abstractions/06_chains/README.md) | Sequential, Map-Reduce, Refine, Router chains |
| 07 | [Memory](./02_core_abstractions/07_memory/README.md) | Buffer, Window, Summary memory, RunnableWithMessageHistory |
| 08 | [Document Loaders](./02_core_abstractions/08_document_loaders/README.md) | PDF, Web, CSV, DOCX, YouTube, Directory loaders |

</details>

<details>
<summary><b>📙 Phase 3 — Retrieval & RAG</b> (5 topics)</summary>

| # | Topic | What You'll Learn |
|---|-------|------------------|
| 09 | [Text Splitters](./03_retrieval_rag/09_text_splitters/README.md) | Recursive, Token, Markdown, Semantic chunking |
| 10 | [Embeddings](./03_retrieval_rag/10_embeddings/README.md) | OpenAI, HuggingFace, Cohere embeddings + caching |
| 11 | [Vector Stores](./03_retrieval_rag/11_vector_stores/README.md) | Chroma, FAISS, Pinecone, metadata filtering |
| 12 | [Retrievers](./03_retrieval_rag/12_retrievers/README.md) | MultiQuery, Compression, Parent Doc, Ensemble |
| 13 | [RAG Patterns](./03_retrieval_rag/13_rag_patterns/README.md) | Naive RAG, HyDE, RAG-Fusion, CRAG, Self-RAG |

</details>

<details>
<summary><b>📕 Phase 4 — Agents & Tools</b> (4 topics)</summary>

| # | Topic | What You'll Learn |
|---|-------|------------------|
| 14 | [Tools & Toolkits](./04_agents_tools/14_tools_toolkits/README.md) | @tool decorator, StructuredTool, built-in tools |
| 15 | [Agents](./04_agents_tools/15_agents/README.md) | ReAct, OpenAI Tools, AgentExecutor, custom agents |
| 16 | [LangGraph](./04_agents_tools/16_langgraph/README.md) | StateGraph, nodes, edges, checkpointing, human-in-loop |
| 17 | [Callbacks & Tracing](./04_agents_tools/17_callbacks_tracing/README.md) | BaseCallbackHandler, token tracking, LangSmith auto-trace |

</details>

<details>
<summary><b>📒 Phase 5 — Production</b> (4 topics)</summary>

| # | Topic | What You'll Learn |
|---|-------|------------------|
| 18 | [LangSmith](./05_production/18_langsmith/README.md) | Tracing, datasets, evaluation, @traceable |
| 19 | [LangServe](./05_production/19_langserve/README.md) | Deploy chains as REST APIs, RemoteRunnable, playground |
| 20 | [Evaluation](./05_production/20_evaluation/README.md) | LangSmith evaluators, RAGAS, custom metrics |
| 21 | [Advanced Patterns](./05_production/21_advanced_patterns/README.md) | Caching, fallbacks, retry, guardrails, cost optimization |

</details>

---

## 🏗️ Repository Structure

```
langchain-learning/
│
├── 01_foundations/
│   ├── 01_introduction/
│   │   ├── README.md          ← Theory & concepts
│   │   ├── examples/
│   │   │   └── 01_basic.py    ← Runnable code
│   │   └── assignment/
│   │       ├── README.md      ← Your task
│   │       └── solution.py    ← Reference solution
│   ├── 02_language_models/
│   ├── 03_prompt_templates/
│   └── 04_output_parsers/
│
├── 02_core_abstractions/      ← LCEL, Chains, Memory, Loaders
├── 03_retrieval_rag/          ← Splitters, Embeddings, VectorStores, RAG
├── 04_agents_tools/           ← Tools, Agents, LangGraph, Callbacks
├── 05_production/             ← LangSmith, LangServe, Evaluation
│
└── 06_projects/               ← End-to-end capstone projects
    ├── 01_chatbot_with_memory/
    ├── 02_rag_private_docs/
    ├── 03_sql_agent/
    └── 04_research_assistant/
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- An OpenAI API key ([get one here](https://platform.openai.com))
- Optional: Anthropic, HuggingFace, or LangSmith API keys

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/langchain-learning.git
cd langchain-learning

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Open .env and fill in your keys
```

### Run your first example

```bash
cd 01_foundations/02_language_models/examples
python 01_basic_llm.py
```

---

## 🔑 API Keys

| Service | Required | Get It |
|---------|----------|--------|
| OpenAI | ✅ Yes | [platform.openai.com](https://platform.openai.com) |
| Anthropic | Optional | [console.anthropic.com](https://console.anthropic.com) |
| LangSmith | Optional (highly recommended) | [smith.langchain.com](https://smith.langchain.com) |
| Tavily Search | Optional | [tavily.com](https://tavily.com) |
| Pinecone | Optional | [pinecone.io](https://pinecone.io) |

---

## 🧩 Core Concepts at a Glance

### LCEL — The Pipe Operator

LangChain Expression Language lets you compose chains elegantly:

```python
# Every component is a Runnable
chain = prompt | llm | parser

# Supports all execution modes
chain.invoke(input)           # single call
chain.batch([i1, i2, i3])    # parallel
chain.stream(input)           # token-by-token streaming
await chain.ainvoke(input)    # async
```

### RAG — Retrieval-Augmented Generation

```python
# Load → Split → Embed → Store → Retrieve → Generate
loader   = PyPDFLoader("document.pdf")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
store    = Chroma.from_documents(splitter.split_documents(loader.load()), embeddings)

chain = (
    {"context": store.as_retriever(), "question": RunnablePassthrough()}
    | rag_prompt | llm | StrOutputParser()
)
answer = chain.invoke("What does the document say about X?")
```

### LangGraph — Stateful Agents

```python
# Define state + nodes + edges → compile → run
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tools", call_tools)
graph.add_conditional_edges("llm", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "llm")   # loop back — agents can cycle!

app = graph.compile(checkpointer=MemorySaver())
result = app.invoke({"messages": [HumanMessage("Search for LangChain news")]})
```

---

## 🎯 Capstone Projects

| Project | Skills Used | Difficulty |
|---------|------------|------------|
| [💬 Chatbot with Memory](./06_projects/01_chatbot_with_memory/) | LLMs, Memory, LCEL | ⭐⭐ |
| [📄 RAG over Private Docs](./06_projects/02_rag_private_docs/) | Loaders, Embeddings, RAG | ⭐⭐⭐ |
| [🗄️ SQL Agent](./06_projects/03_sql_agent/) | Tools, Agents, LangGraph | ⭐⭐⭐ |
| [🔬 Research Assistant](./06_projects/04_research_assistant/) | Multi-agent, LangGraph | ⭐⭐⭐⭐ |

---

## 🛠️ Tech Stack

```
langchain >= 0.3.0          Core framework
langchain-openai             GPT-4o, embeddings
langchain-community          100+ integrations
langgraph                    Agent orchestration
langserve                    API deployment
langsmith                    Observability
chromadb                     Local vector store
faiss-cpu                    Fast similarity search
python-dotenv                Env management
pydantic >= 2.0              Data validation
```

---

## 📈 Learning Path

```
Week 1 ──► Foundations      (Topics 01–04)   Master LLMs, prompts, parsers
Week 2 ──► Abstractions     (Topics 05–08)   LCEL, Chains, Memory, Loaders
Week 3 ──► RAG              (Topics 09–13)   Build a document Q&A system
Week 4 ──► Agents           (Topics 14–17)   Build a tool-calling agent
Week 5 ──► Production       (Topics 18–21)   Deploy and monitor your app
Week 6 ──► Projects         (06_projects)    Ship 4 real applications
```

---

## 🤝 Contributing

Contributions are welcome! To contribute:

```bash
# Fork the repo, then:
git checkout -b feature/your-topic
# Make your changes
git commit -m "Add: topic explanation for XYZ"
git push origin feature/your-topic
# Open a Pull Request
```

Great contribution ideas:
- Fix errors or outdated code examples
- Add missing topic READMEs or examples
- Improve assignment prompts
- Add new project ideas

---

## 📚 Resources

| Resource | Link |
|---------|------|
| Official LangChain Docs | [python.langchain.com](https://python.langchain.com) |
| LangGraph Docs | [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph) |
| LangSmith | [smith.langchain.com](https://smith.langchain.com) |
| LangChain GitHub | [github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain) |
| LangChain YouTube | [youtube.com/@LangChain](https://youtube.com/@LangChain) |
| LangChain Discord | [discord.gg/langchain](https://discord.gg/langchain) |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with 🦜 while learning LangChain from scratch**

*Star ⭐ this repo if it helped you!*

[![GitHub](https://img.shields.io/badge/GitHub-hiralchudasam-181717?style=flat-square&logo=github)](https://github.com/hiralchudasam)

</div>
