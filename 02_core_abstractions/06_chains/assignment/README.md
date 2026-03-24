# Assignment 06 — Chains

## 🎯 Task
Build a **study quiz generator** using sequential chains:

1. **Chain 1 (Topic → Questions):** Given a topic and difficulty level, generate 5 quiz questions
2. **Chain 2 (Questions → Answers):** For each question, generate a model answer
3. **Chain 3 (Q+A → Quiz):** Format everything into a clean printable quiz

## Requirements
- Connect chains sequentially (output of one → input of next)
- Support `difficulty`: easy / medium / hard
- Test with at least 3 different topics

## Bonus
- Add a map-reduce step: generate questions per subtopic, then combine
- Export the quiz to a `.txt` file

## Reference Solution
See `solution.py` after attempting it yourself!
