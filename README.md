# DocuMind-RAG: Document Q&A Assistant

🎥 **Demo Video:** https://www.loom.com/share/bfd8458b0f434d9da04bca28312a3e3a

A simple RAG-based application that lets you upload documents and ask questions about them.  
It retrieves relevant context from the document and generates grounded answers with source references.

Built as a lightweight end-to-end system using FastAPI, TF-IDF retrieval, and an LLM.

## Features

1. Upload `.pdf` or `.docx`
2. Extract text (`pdfplumber`, `python-docx`)
3. Chunk text into smaller pieces
4. Embed chunks with `text-embedding-3-small`
5. Ask questions
6. Retrieve most relevant chunks via cosine similarity
7. Answer with `gpt-4.1-mini` using only retrieved context
8. Return answer and source chunks

## Project Structure

```text
.
├── main.py
├── requirements.txt
└── static/
    └── index.html
```

## Setup

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set API provider and key.

Option A: OpenAI (default)

```bash
# Linux/macOS
export OPENAI_API_KEY="your_api_key_here"

# Windows PowerShell
$env:OPENAI_API_KEY="your_api_key_here"
```

Option B: Google AI Studio (Gemini via OpenAI-compatible endpoint)

```bash
# Linux/macOS
export LLM_PROVIDER="gemini"
export GEMINI_API_KEY="your_gemini_api_key_here"

# Windows PowerShell
$env:LLM_PROVIDER="gemini"
$env:GEMINI_API_KEY="your_gemini_api_key_here"
```

Optional model overrides:
- `EMBEDDING_MODEL`
- `CHAT_MODEL`

Defaults:
- OpenAI: `text-embedding-3-small` + `gpt-4.1-mini`
- Gemini: `gemini-embedding-001` + `gemini-2.5-flash`

4. Run the server:

```bash
uvicorn main:app --reload
```

5. Open:

```text
http://127.0.0.1:8000
```

## API Endpoints

### `POST /upload`
- Form field: `file` (PDF or DOCX)
- Processes document and stores chunk embeddings in memory.

### `POST /ask`
- JSON body:

```json
{
  "question": "What does the document say about ...?"
}
```

- Response:

```json
{
  "answer": "...",
  "sources": ["chunk text 1", "chunk text 2", "chunk text 3"],
  "document": "filename.pdf"
}
```

## Notes

- This is intentionally minimal and keeps only one active uploaded document at a time.
- The LLM receives only retrieved chunks, not the full document.
