# Assignment 16 — LangGraph

## 🎯 Goal

Build a **Research Assistant Agent** using LangGraph that:
1. Takes a research question from the user
2. Searches the web for information (Tavily or DuckDuckGo)
3. Synthesizes a structured research report
4. Can ask clarifying questions if the query is too vague

---

## Requirements

Build a LangGraph with these nodes:
- `classify` — determines if the query is clear or needs clarification
- `clarify` — asks the user a follow-up question (human-in-the-loop)
- `search` — performs web search using a tool
- `synthesize` — writes a structured report from search results
- `END` — outputs the final report

```
START → classify → (clear?) → search → synthesize → END
                 → (vague?) → clarify → search → synthesize → END
```

---

## Expected Output

```
🔍 Research Assistant

Your question: AI in healthcare 2024

[classify] Query is clear, proceeding to search...
[search] Searching: "AI healthcare applications 2024"
[search] Searching: "AI diagnostics hospitals 2024"
[synthesize] Writing report...

📋 RESEARCH REPORT
==================
Topic: AI in Healthcare 2024

Summary: [2-3 paragraph report...]

Key Findings:
  • Finding 1
  • Finding 2
  • Finding 3

Sources:
  1. https://...
  2. https://...
```

---

## Bonus Challenges

- [ ] Save the report to a `.md` file
- [ ] Add a `fact_check` node that verifies key claims
- [ ] Support multiple searches (agent decides how many to run)
- [ ] Add a confidence score to the final report

---

## Hints

- Use `interrupt_before=["clarify"]` for human-in-the-loop
- Use `MemorySaver` for checkpointing between turns
- Store search results in the state
- Use `RunnableLambda` for the synthesis step

---

## Reference Solution

See [`solution.py`](./solution.py) after attempting it yourself!
