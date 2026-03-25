# 09 — Text Splitters

## Why Split Text?

LLMs have a **context window limit** — they can't process an entire book in one call. Text splitters break large documents into smaller, overlapping **chunks** that fit within token limits and can be embedded and retrieved individually.

```
Large Document (100,000 tokens)
         ↓ Split
[chunk_1 | chunk_2 | chunk_3 | ... | chunk_n]  (each ~500-1000 tokens)
         ↓ Embed
Vector Store → Semantic Search
```

---

## Key Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `chunk_size` | 1000 | Target size of each chunk (in chars or tokens) |
| `chunk_overlap` | 200 | How many chars/tokens overlap between chunks |
| `length_function` | `len` | How to measure chunk size |
| `separators` | varies | Characters to split on (in priority order) |

**Overlap** prevents losing context at chunk boundaries. If a sentence spans two chunks, both chunks will contain it.

---

## Splitter Types

### RecursiveCharacterTextSplitter (recommended default)
Splits on a priority list of separators: `\n\n` → `\n` → `. ` → ` ` → `""`.
Works well for most plain text.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = splitter.split_text(text)
# or
chunks = splitter.split_documents(documents)
```

### CharacterTextSplitter
Splits only on a single separator (default: `\n\n`). Simpler but less smart.

### TokenTextSplitter
Splits by actual token count (using tiktoken). Use when you care about token limits precisely.

```python
from langchain.text_splitter import TokenTextSplitter
splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
```

### MarkdownHeaderTextSplitter
Splits on markdown headers, preserving document structure in metadata.

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
chunks = splitter.split_text(markdown_text)
# Each chunk has metadata: {"h1": "Chapter 1", "h2": "Section A"}
```

### HTMLHeaderTextSplitter
Same idea but for HTML documents.

### Semantic Chunking
Uses embeddings to split at "semantic boundaries" rather than fixed character counts. More intelligent but slower.

```python
from langchain_experimental.text_splitter import SemanticChunker
splitter = SemanticChunker(embeddings=OpenAIEmbeddings())
chunks = splitter.split_text(text)
```

---

## Choosing Chunk Size

| Content Type | chunk_size | chunk_overlap |
|-------------|-----------|--------------|
| General text | 1000 | 200 |
| Legal/Technical docs | 500 | 100 |
| Code | 2000 | 400 |
| Q&A pairs | 200-300 | 0 |

**Rule of thumb:** chunk_overlap ≈ 10-20% of chunk_size.

---

## Next Topic

→ [10 — Embeddings](../10_embeddings/README.md)
