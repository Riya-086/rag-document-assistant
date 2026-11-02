import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .answer import NOT_FOUND, ClaudeAnswerer, extractive_answer
from .extract import UnsupportedFile, extract_text
from .search import BM25Index
from .store import DocumentStore
from .text import chunk

logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
STATIC = Path(__file__).parent / "static"
SAMPLES = Path(__file__).resolve().parent.parent / "sample-docs"


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=4, ge=1, le=8)


def create_app(db_path: str = ":memory:", answerer=None, load_samples: bool = False) -> FastAPI:
    app = FastAPI(title="RAG Document Assistant", version="1.0.0")
    store, index = DocumentStore(db_path), BM25Index()
    for c in store.all_chunks():
        index.add(c)

    def ingest(filename: str, text: str) -> dict:
        pieces = chunk(text)
        if not pieces:
            raise HTTPException(422, "The file has no readable text.")
        doc, chunks = store.add(filename, pieces)
        for c in chunks:
            index.add(c)
        return doc

    if load_samples and not store.documents():
        for path in sorted(SAMPLES.glob("*.md")):
            ingest(path.name, path.read_text(encoding="utf-8"))

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/documents")
    def list_documents():
        return store.documents()

    @app.post("/api/documents", status_code=201)
    async def upload(file: UploadFile):
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Files can be at most 2 MB.")
        try:
            text = extract_text(file.filename or "", data)
        except UnsupportedFile as exc:
            raise HTTPException(415, str(exc)) from exc
        return ingest(Path(file.filename or "document").name, text)

    @app.delete("/api/documents/{doc_id}", status_code=204)
    def delete(doc_id: int):
        if not store.delete(doc_id):
            raise HTTPException(404, "Document not found.")
        index.remove_document(doc_id)

    @app.post("/api/ask")
    def ask(body: Question):
        hits = index.search(body.question, k=body.top_k)
        sources = [
            {"n": n, "document": h.chunk.document, "score": h.score, "text": h.chunk.text}
            for n, h in enumerate(hits, start=1)
        ]
        if not hits:  # guardrail: never ask a model to answer without evidence
            return {"answer": NOT_FOUND, "grounded": False, "mode": "none", "sources": []}
        if answerer is not None:
            try:
                answer = answerer(body.question, hits)
                return {"answer": answer, "grounded": answer != NOT_FOUND, "mode": "llm", "sources": sources}
            except Exception:
                logger.exception("LLM call failed; falling back to the extractive answer")
        answer = extractive_answer(body.question, hits)
        return {"answer": answer, "grounded": answer != NOT_FOUND, "mode": "extractive", "sources": sources}

    return app


def _answerer_from_env():
    key = os.environ.get("ANTHROPIC_API_KEY")
    return ClaudeAnswerer(key, os.environ.get("LLM_MODEL", "claude-haiku-5-5")) if key else None


app = (
    create_app(
        db_path=os.environ.get("DATABASE_PATH", "documents.db"),
        answerer=_answerer_from_env(),
        load_samples=os.environ.get("LOAD_SAMPLE_DOCS", "1") == "1",
    )
    if os.environ.get("RAG_SKIP_DEFAULT_APP") != "1"
    else None
)
