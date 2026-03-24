# Assignment 05 — LCEL

## 🎯 Task
Build a **blog post generator** using LCEL that:
1. Takes a `topic` and `audience` as input
2. Uses `RunnableParallel` to simultaneously generate: title, outline (3 sections), and intro paragraph
3. Combines everything into a formatted blog draft
4. Streams the final draft to the console

## Requirements
- Use `RunnableParallel` for simultaneous generation
- Use `RunnablePassthrough` to pass original inputs
- Use `RunnableLambda` to format/merge the parallel outputs
- Stream the final output

## Bonus
- Add a `tone` parameter (formal/casual/technical)
- Calculate and display total generation time

## Reference Solution
See `solution.py` after attempting it yourself!
