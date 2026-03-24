# Assignment 08 — Document Loaders

## 🎯 Task
Build a **document ingestion pipeline** that:
1. Loads all `.txt` files from a given folder using `DirectoryLoader`
2. For each document, extracts: filename, word count, char count, first sentence
3. Prints a summary table
4. Uses an LLM to generate a one-line description of each document's content

## Requirements
- Use `DirectoryLoader` + `TextLoader`
- Inspect `doc.metadata` for source info
- Test with at least 3 sample text files (create them yourself)

## Bonus
- Support PDF files using `PyPDFLoader`
- Sort documents by word count
- Save the summary to a CSV file

## Reference Solution
See `solution.py` after attempting it yourself!
