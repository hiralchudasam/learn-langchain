# 13 — RAG Patterns

## What is RAG?

**Retrieval-Augmented Generation** — instead of asking the LLM from memory, you first retrieve relevant documents from a knowledge base, then ask the LLM to answer using those documents.

```
User Question
     ↓
[Retriever] → Relevant Docs
     ↓
[LLM] + Docs → Grounded Answer
```

---

## Why RAG?

| Problem | Without RAG | With RAG |
|---------|------------|---------|
| Knowledge cutoff | LLM doesn't know recent events | Can use up-to-date docs |
| Private data | LLM never saw your docs | Retrieves from your database |
| Hallucination | LLM makes things up | Grounded in real sources |
| Source citations | Not possible | Can cite which doc was used |

---

## Naive RAG Pipeline

The simplest RAG: load → split → embed → store → retrieve → generate.

```python
# 1. Load
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# 2. Split
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 3. Embed + Store
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())

# 4. Retrieve + Generate
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

---

## Advanced RAG Patterns

### HyDE (Hypothetical Document Embeddings)
Generate a *hypothetical* answer, embed it, and use that for retrieval.
Good for: short or vague queries that don't embed well.

```
Query → LLM generates fake answer → embed fake answer → retrieve real docs
```

### MultiQuery Retrieval
Generate multiple versions of the query, retrieve for each, deduplicate.

```
Query → LLM generates 3 variants → retrieve for each → union of results
```

### RAG-Fusion
MultiQuery + Reciprocal Rank Fusion for ranking.

### Corrective RAG (CRAG)
After retrieval, evaluate relevance. If docs are bad, fall back to web search.

### Self-RAG
LLM decides *when* to retrieve (not always), and evaluates its own answer.

### Parent Document Retrieval
Store large chunks but embed small chunks. Retrieve small, return parent.

---

## Evaluation Metrics

| Metric | Measures |
|--------|---------|
| Context precision | Are retrieved docs relevant? |
| Context recall | Were all relevant docs retrieved? |
| Answer faithfulness | Is answer grounded in context? |
| Answer relevancy | Does answer address the question? |

Use **RAGAS** framework for automated evaluation.

---

## RAG Prompt Template

```python
RAG_TEMPLATE = """Use the following context to answer the question.
If you don't know the answer from the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""
```

---

## Next Topic

→ [14 — Tools & Toolkits](../../04_agents_tools/14_tools_toolkits/README.md)
