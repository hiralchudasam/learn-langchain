# Assignment 03 — Prompt Templates

## 🎯 Task
Build a **multi-language code reviewer** using ChatPromptTemplate.

The system should accept:
- `code` — the code to review
- `language` — programming language (Python, JS, SQL, etc.)
- `review_type` — one of: `style`, `security`, `performance`

And return targeted review comments based on the review type.

## Requirements
- Use `ChatPromptTemplate` with a system + human message
- Use `PartialPromptTemplate` to create pre-set reviewer chains (e.g., `python_security_reviewer`)
- Test with at least 3 different code snippets

## Bonus
- Add a `severity` field (info/warning/critical) to each comment
- Use `PydanticOutputParser` to return structured feedback

## Reference Solution
See `solution.py` after attempting it yourself!
