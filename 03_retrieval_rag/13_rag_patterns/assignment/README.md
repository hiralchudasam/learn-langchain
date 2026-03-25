# Assignment 13 — RAG Patterns

## 🎯 Goal

Build a **"Chat with your PDF"** application using a full RAG pipeline with conversational memory.

---

## Requirements

1. Accept a PDF file path as input
2. Load and split it using `RecursiveCharacterTextSplitter`
3. Embed and store chunks in **Chroma** (persisted to disk)
4. Build a **conversational RAG chain** that:
   - Reformulates the question based on conversation history
   - Retrieves relevant chunks
   - Generates a grounded answer with source page numbers
5. Run a CLI loop where the user can ask questions about the PDF

---

## Expected Output

```
📄 Loading: ml_research_paper.pdf
✂️  Split into 47 chunks
🔢 Generating embeddings...
✅ Vector store ready!

💬 Ask questions about your document (type 'exit' to quit)

You: What is the main contribution of this paper?
Bot: [answer grounded in the PDF...]
Sources: pages 1, 3

You: Can you elaborate on the methodology?
Bot: [follows up correctly using history...]

You: exit
```

---

## Bonus Challenges

- [ ] Support multiple PDF files (upload a folder)
- [ ] Show which chunk/page each answer came from
- [ ] Add a `summarize` command that summarizes the whole document
- [ ] Cache embeddings so re-running doesn't re-embed
- [ ] Support `.txt` and `.docx` files in addition to PDFs

---

## Hints

- Use `PyPDFLoader` for PDF loading (includes page metadata)
- Use `MessagesPlaceholder` + a reformulation chain for follow-up questions
- Access page numbers from `doc.metadata["page"]`
- Persist Chroma with `persist_directory` so you don't re-embed on every run

---

## Reference Solution

See [`solution.py`](./solution.py) after attempting it yourself!
