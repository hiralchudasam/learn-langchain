# Assignment 01 — Introduction to LangChain

## 🎯 Task
Build a CLI "LangChain Concept Explorer" that:
1. Asks the user to enter any LangChain concept (e.g. "LCEL", "RAG", "LangGraph")
2. Uses ChatOpenAI to return a structured explanation with: definition, how it works, use case, and a mini code snippet
3. Loops until the user types "exit"

## Requirements
- Use `ChatOpenAI` with a system prompt
- Format output clearly (use headers/sections)
- Handle unknown concepts gracefully

## Bonus
- Cache results so the same concept isn't re-fetched
- Add a `/list` command showing all concepts explored in the session

## Reference Solution
See `solution.py` after attempting it yourself!
