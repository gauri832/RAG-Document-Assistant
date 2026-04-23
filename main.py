import io
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pdfplumber
from docx import Document
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel


app = FastAPI(title="Document Q&A Assistant (RAG)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


class AskRequest(BaseModel):
    question: str


@dataclass
class ChunkRecord:
    text: str
    embedding: np.ndarray


chunks_store: List[ChunkRecord] = []
embedding_matrix: np.ndarray | None = None
embedding_norms: np.ndarray | None = None
current_filename: str | None = None


def _detect_provider(default_provider: str) -> str:
    """
    Auto-detect provider when user forgets to set LLM_PROVIDER.
    """
    provider = default_provider.strip().lower()
    if provider != "openai":
        return provider

    maybe_google_key = os.getenv("OPENAI_API_KEY", "")
    if maybe_google_key.startswith("AIza"):
        return "gemini"
    return "openai"


def get_client_and_models() -> tuple[OpenAI, str, str]:
    """
    Returns (client, embedding_model, chat_model) based on env config.
    """
    provider = _detect_provider(os.getenv("LLM_PROVIDER", "openai"))

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Set OPENAI_API_KEY.")
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        chat_model = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
        return OpenAI(api_key=api_key, timeout=60.0, max_retries=2), embedding_model, chat_model

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Set GEMINI_API_KEY.")
        embedding_model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        chat_model = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=60.0,
            max_retries=2,
        )
        return client, embedding_model, chat_model

    raise HTTPException(status_code=500, detail="Invalid LLM_PROVIDER. Use openai or gemini.")


def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = os.path.splitext(filename.lower())[1]

    if ext == ".pdf":
        text_parts: List[str] = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                page_text = page_text.strip()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts).strip()

    if ext == ".docx":
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs).strip()

    raise HTTPException(status_code=400, detail="Only PDF and DOCX are supported.")


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> List[str]:
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks: List[str] = []
    start = 0
    total = len(words)
    while start < total:
        end = min(start + chunk_size, total)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == total:
            break
        start = end - overlap
    return chunks


def embed_texts(client: OpenAI, texts: List[str], embedding_model: str) -> List[np.ndarray]:
    if not texts:
        return []

    out: List[np.ndarray] = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            resp = client.embeddings.create(model=embedding_model, input=batch)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Embedding API error: {exc}") from exc

        for item in resp.data:
            out.append(np.array(item.embedding, dtype=np.float32))
    return out


def embed_query(client: OpenAI, question: str, embedding_model: str) -> np.ndarray:
    try:
        resp = client.embeddings.create(model=embedding_model, input=question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Question embedding error: {exc}") from exc
    return np.array(resp.data[0].embedding, dtype=np.float32)


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global chunks_store, embedding_matrix, embedding_norms, current_filename

    filename = file.filename or "uploaded_file"
    ext = os.path.splitext(filename.lower())[1]
    if ext not in [".pdf", ".docx"]:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    text = extract_text(content, filename)
    if not text:
        raise HTTPException(status_code=400, detail="No readable text found in document.")

    split_chunks = chunk_text(text, chunk_size=400, overlap=60)
    if not split_chunks:
        raise HTTPException(status_code=400, detail="Could not create chunks from document.")

    client, embedding_model, _ = get_client_and_models()
    vectors = embed_texts(client, split_chunks, embedding_model)

    if len(vectors) != len(split_chunks):
        raise HTTPException(status_code=500, detail="Embedding count mismatch.")

    chunks_store = [ChunkRecord(text=t, embedding=v) for t, v in zip(split_chunks, vectors)]
    embedding_matrix = np.vstack(vectors)
    embedding_norms = np.linalg.norm(embedding_matrix, axis=1)
    current_filename = filename

    return {
        "message": "Document processed successfully.",
        "filename": current_filename,
        "chunks": len(chunks_store),
    }


@app.post("/ask")
async def ask_question(payload: AskRequest):
    if not chunks_store or embedding_matrix is None or embedding_norms is None:
        raise HTTPException(status_code=400, detail="Upload a document first.")

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    client, embedding_model, chat_model = get_client_and_models()
    q_vec = embed_query(client, question, embedding_model)

    q_norm = float(np.linalg.norm(q_vec))
    if q_norm == 0:
        raise HTTPException(status_code=400, detail="Could not embed question.")

    # Fast cosine similarity against all chunks.
    denom = (embedding_norms * q_norm) + 1e-10
    scores = (embedding_matrix @ q_vec) / denom

    top_k = min(3, len(scores))
    top_idx = np.argsort(scores)[-top_k:][::-1]
    top_chunks = [chunks_store[int(i)].text for i in top_idx]
    context = "\n\n---\n\n".join(top_chunks)

    try:
        completion = client.chat.completions.create(
            model=chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer ONLY using the provided context. "
                        "If the answer is not in the context, say you don't know."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{question}",
                },
            ],
            temperature=0,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM API error: {exc}") from exc

    answer = completion.choices[0].message.content or "I don't know."
    return {
        "answer": answer.strip(),
        "sources": top_chunks,
        "document": current_filename,
    }
