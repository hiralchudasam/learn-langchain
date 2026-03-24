# 07 — Memory

## What is Memory in LangChain?

By default, LLMs are stateless — every call is independent. Memory adds conversation history so the model can remember previous turns.

## Memory Types

| Type | Keeps | Best For |
|------|-------|---------|
| `ConversationBufferMemory` | Full history | Short conversations |
| `ConversationBufferWindowMemory` | Last N turns | Medium chats |
| `ConversationSummaryMemory` | Compressed summary | Long conversations |
| `ConversationSummaryBufferMemory` | Summary + recent | Production chatbots |
| `VectorStoreRetrieverMemory` | Relevant past messages | Very long / topic-specific |
| `EntityMemory` | Named entities | Character/name tracking |

## Modern Approach (LCEL)

With LCEL, you manage history yourself using `MessagesPlaceholder`:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

history = []

def chat(user_input: str) -> str:
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])
    chain = template | llm | StrOutputParser()
    response = chain.invoke({"input": user_input, "history": history})
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response))
    return response
```

## RunnableWithMessageHistory

LangChain provides `RunnableWithMessageHistory` to automate this pattern with persistent session storage.

## Next Topic
→ [08 — Document Loaders](../08_document_loaders/README.md)
