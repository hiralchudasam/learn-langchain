"""
13 — RAG Patterns
Example 01: Naive RAG, MultiQuery RAG, Conversational RAG
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()

# ─── Sample Documents ─────────────────────────────────────────────────────────

SAMPLE_DOCS = [
    Document(page_content="""
    LangChain is an open-source framework for building LLM-powered applications.
    It provides abstractions for prompts, models, memory, and tools.
    LangChain was created by Harrison Chase in 2022.
    The latest version supports LCEL (LangChain Expression Language) for composing chains.
    """, metadata={"source": "langchain_overview.txt"}),

    Document(page_content="""
    RAG stands for Retrieval-Augmented Generation. It combines information retrieval
    with language generation. In RAG, relevant documents are retrieved from a knowledge base
    and provided as context to the LLM before it generates a response.
    RAG reduces hallucination and allows LLMs to answer questions about private data.
    """, metadata={"source": "rag_overview.txt"}),

    Document(page_content="""
    Vector databases store embeddings — numerical representations of text.
    Popular vector databases include Chroma, Pinecone, Weaviate, and Qdrant.
    Semantic search finds documents whose meaning is similar to the query,
    even if they don't share exact keywords. This is done by comparing embedding vectors.
    """, metadata={"source": "vector_stores.txt"}),

    Document(page_content="""
    LangGraph is a library for building stateful multi-agent applications.
    It extends LangChain with graph-based orchestration, allowing cycles and conditionals.
    LangGraph supports human-in-the-loop workflows and persistent checkpointing.
    It is built on top of LangChain core and integrates with LangSmith for tracing.
    """, metadata={"source": "langgraph.txt"}),
]

# ─── 1. Naive RAG ─────────────────────────────────────────────────────────────

print("=" * 55)
print("1. Naive RAG Pipeline")
print("=" * 55)

# Build vector store from sample docs
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(SAMPLE_DOCS)

vectorstore = Chroma.from_documents(chunks, embeddings, collection_name="langchain_demo")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an assistant for question-answering tasks.
Use the following context to answer the question.
If you don't know the answer from the context, say "I don't know."
Keep the answer concise.

Context:
{context}"""),
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n".join(f"[{d.metadata.get('source','?')}]\n{d.page_content}" for d in docs)

naive_rag_chain = (
    RunnableParallel(
        context=(retriever | format_docs),
        question=RunnablePassthrough(),
    )
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

questions = [
    "What is LangChain?",
    "How does RAG reduce hallucination?",
    "What is LangGraph used for?",
    "What is the price of gold?",  # Not in docs → should say "I don't know"
]

for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {naive_rag_chain.invoke(q)}")

# ─── 2. RAG with Source Citations ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("2. RAG with Source Citations")
print("=" * 55)

CITATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Answer using the context. End your answer with the sources used.

Context:
{context}"""),
    ("human", "{question}"),
])

def retrieve_with_metadata(question: str):
    docs = retriever.invoke(question)
    context = format_docs(docs)
    sources = list({d.metadata.get("source", "unknown") for d in docs})
    return {"context": context, "sources": sources}

def rag_with_citations(question: str):
    retrieved = retrieve_with_metadata(question)
    answer = (CITATION_PROMPT | llm | StrOutputParser()).invoke({
        "context": retrieved["context"],
        "question": question,
    })
    return answer, retrieved["sources"]

answer, sources = rag_with_citations("What is RAG?")
print(f"Q: What is RAG?")
print(f"A: {answer}")
print(f"Sources: {sources}")

# ─── 3. MultiQuery RAG ────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("3. MultiQuery RAG")
print("=" * 55)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm,
)

# MultiQueryRetriever generates multiple query variants automatically
docs = multi_query_retriever.invoke("tell me about the vector db tools")
print(f"Retrieved {len(docs)} unique docs via MultiQuery")
for d in docs:
    print(f"  - {d.metadata.get('source')}: {d.page_content[:60]}...")

# ─── 4. Conversational RAG ────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("4. Conversational RAG (with history)")
print("=" * 55)

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Given chat history and the latest user question,
reformulate the question to be standalone (no references to history).
If already standalone, return as is. Do NOT answer it."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Answer the question using the context below.
If you don't know, say so.

{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

contextualize_chain = CONTEXTUALIZE_PROMPT | llm | StrOutputParser()

chat_history = []

def conversational_rag(user_input: str) -> str:
    # Step 1: Reformulate question considering history
    if chat_history:
        standalone_q = contextualize_chain.invoke({
            "input": user_input,
            "chat_history": chat_history,
        })
    else:
        standalone_q = user_input

    # Step 2: Retrieve
    docs = retriever.invoke(standalone_q)
    context = format_docs(docs)

    # Step 3: Answer
    answer = (QA_PROMPT | llm | StrOutputParser()).invoke({
        "input": user_input,
        "context": context,
        "chat_history": chat_history,
    })

    # Step 4: Update history
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=answer))

    return answer

print("Turn 1:", conversational_rag("What is LangChain?"))
print("\nTurn 2:", conversational_rag("When was it created?"))     # refers to LangChain
print("\nTurn 3:", conversational_rag("Does it support agents?"))   # still about LangChain
