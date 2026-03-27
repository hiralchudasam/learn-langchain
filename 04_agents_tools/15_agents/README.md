# 15 — Agents

## What is an Agent?

An agent uses an LLM as a reasoning engine to decide which tools to call, in what order, to accomplish a task. Unlike a fixed chain, an agent's execution path is dynamic.

```
User Question
    ↓
LLM thinks: "I need to search for X, then calculate Y"
    ↓
Calls search tool → gets result
    ↓
Calls calculator → gets result
    ↓
LLM synthesizes final answer
```

## ReAct Pattern (Reason + Act)

The classic agent loop:
1. **Reason** — LLM thinks about what to do
2. **Act** — LLM calls a tool
3. **Observe** — LLM sees the tool result
4. **Repeat** until it has enough information
5. **Answer** — LLM produces the final response

## Modern: create_react_agent (LangGraph)

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(llm, tools)
result = agent.invoke({"messages": [HumanMessage("Search for LangChain news")]})
```

## Legacy: AgentExecutor

```python
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain import hub

prompt = hub.pull("hwchase17/openai-tools-agent")
agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = executor.invoke({"input": "What is the weather in Mumbai?"})
```

## Agent Types

| Agent | Model Requirement | Notes |
|-------|------------------|-------|
| ReAct | Any chat model | Most universal |
| OpenAI Tools | OpenAI function calling | Best with GPT-4 |
| XML Agent | Good instruction following | Claude-compatible |

## Key Agent Parameters

```python
AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,         # stop after N steps
    handle_parsing_errors=True, # retry on bad output
    verbose=True,              # print each step
)
```

## Next Topic
→ [16 — LangGraph](../16_langgraph/README.md)
