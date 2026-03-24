# 08 — Document Loaders

## What are Document Loaders?

Document loaders read files from various sources and convert them into LangChain `Document` objects — each with `page_content` (text) and `metadata` (source, page number, etc.).

## Common Loaders

### Text & PDF
```python
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader

loader = PyPDFLoader("report.pdf")  # page_content per page, metadata includes "page"
loader = TextLoader("notes.txt")
loader = CSVLoader("data.csv")
docs = loader.load()
```

### Web
```python
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(["https://python.org", "https://langchain.com"])
docs = loader.load()
```

### Unstructured (universal loader)
```python
from langchain_community.document_loaders import UnstructuredFileLoader
loader = UnstructuredFileLoader("file.docx")  # handles PDF, DOCX, HTML, etc.
```

### YouTube
```python
from langchain_community.document_loaders import YoutubeLoader
loader = YoutubeLoader.from_youtube_url("https://youtube.com/watch?v=...")
docs = loader.load()  # transcripts as documents
```

### Directory Loader (load all files in a folder)
```python
from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader("./docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()
```

## Document Structure
```python
doc = docs[0]
doc.page_content   # The text
doc.metadata       # {"source": "report.pdf", "page": 0, ...}
```

## Lazy Loading (memory efficient)
```python
for doc in loader.lazy_load():   # one doc at a time
    process(doc)
```

## Next Topic
→ [09 — Text Splitters](../../03_retrieval_rag/09_text_splitters/README.md)
