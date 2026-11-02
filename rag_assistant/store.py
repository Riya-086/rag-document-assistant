import sqlite3
from datetime import UTC, datetime

from .search import Chunk


class DocumentStore:
    def __init__(self, path: str):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY, filename TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
                position INTEGER NOT NULL, text TEXT NOT NULL);
        """)

    def add(self, filename: str, chunks: list[str]) -> tuple[dict, list[Chunk]]:
        with self.db:
            now = datetime.now(UTC).isoformat(timespec="seconds")
            doc_id = self.db.execute(
                "INSERT INTO documents (filename, created_at) VALUES (?, ?)", (filename, now)
            ).lastrowid
            stored = []
            for position, text in enumerate(chunks):
                chunk_id = self.db.execute(
                    "INSERT INTO chunks (document_id, position, text) VALUES (?, ?, ?)", (doc_id, position, text)
                ).lastrowid
                stored.append(Chunk(chunk_id, doc_id, filename, text))
        return {"id": doc_id, "filename": filename, "createdAt": now, "chunks": len(chunks)}, stored

    def documents(self) -> list[dict]:
        rows = self.db.execute("""
            SELECT d.id, d.filename, d.created_at, COUNT(c.id) AS chunks
            FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id ORDER BY d.id""").fetchall()
        return [
            {"id": r["id"], "filename": r["filename"], "createdAt": r["created_at"], "chunks": r["chunks"]}
            for r in rows
        ]

    def delete(self, doc_id: int) -> bool:
        with self.db:
            return self.db.execute("DELETE FROM documents WHERE id = ?", (doc_id,)).rowcount > 0

    def all_chunks(self) -> list[Chunk]:
        rows = self.db.execute("""
            SELECT c.id, c.document_id, d.filename, c.text FROM chunks c
            JOIN documents d ON d.id = c.document_id ORDER BY c.id""").fetchall()
        return [Chunk(r["id"], r["document_id"], r["filename"], r["text"]) for r in rows]
