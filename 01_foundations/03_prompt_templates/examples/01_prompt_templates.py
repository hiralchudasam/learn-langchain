"""
03 — Prompt Templates
Example 01: PromptTemplate, ChatPromptTemplate, FewShot, Partial
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ─── 1. Basic PromptTemplate ──────────────────────────────────────────────────

template = PromptTemplate.from_template("Tell me a {adjective} joke about {topic}.")
prompt = template.invoke({"adjective": "funny", "topic": "programmers"})
print("PromptTemplate:", prompt.text)

# ─── 2. ChatPromptTemplate ────────────────────────────────────────────────────

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {domain} teacher. Be concise."),
    ("human", "Explain {concept} in simple terms."),
])

messages = chat_template.invoke({"domain": "AI", "concept": "embeddings"})
print("\nChatPromptTemplate messages:")
for m in messages.messages:
    print(f"  [{m.__class__.__name__}]: {m.content}")

response = (chat_template | llm | StrOutputParser()).invoke({
    "domain": "AI", "concept": "embeddings"
})
print("\nResponse:", response)

# ─── 3. MessagesPlaceholder (for conversation history) ────────────────────────

from langchain_core.messages import HumanMessage, AIMessage

history_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

history = [
    HumanMessage(content="My favorite color is blue."),
    AIMessage(content="That's a great color! Blue is calming and serene."),
]

response = (history_template | llm | StrOutputParser()).invoke({
    "history": history,
    "input": "What is my favorite color?",
})
print("\nWith history:", response)

# ─── 4. Partial Templates ─────────────────────────────────────────────────────

base_template = ChatPromptTemplate.from_messages([
    ("system", "You translate text to {language}."),
    ("human", "{text}"),
])

spanish_chain = base_template.partial(language="Spanish") | llm | StrOutputParser()
french_chain  = base_template.partial(language="French")  | llm | StrOutputParser()

print("\nSpanish:", spanish_chain.invoke({"text": "Hello, how are you?"}))
print("French:", french_chain.invoke({"text": "Hello, how are you?"}))

# ─── 5. FewShotPromptTemplate ─────────────────────────────────────────────────

examples = [
    {"input": "happy",  "output": "sad"},
    {"input": "tall",   "output": "short"},
    {"input": "fast",   "output": "slow"},
]

example_prompt = PromptTemplate.from_template("Word: {input} → Antonym: {output}")

few_shot = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Give the antonym of each word:",
    suffix="Word: {input} → Antonym:",
    input_variables=["input"],
)

prompt_str = few_shot.format(input="bright")
print("\nFewShot prompt:\n", prompt_str)

response = (few_shot | llm | StrOutputParser()).invoke({"input": "bright"})
print("FewShot response:", response)
