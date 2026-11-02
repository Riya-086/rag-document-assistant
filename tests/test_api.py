from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from rag_assistant.answer import NOT_FOUND, ClaudeAnswerer
from rag_assistant.api import create_app

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def client():
    return TestClient(create_app(load_samples=True))


def ask(client, question):
    res = client.post("/api/ask", json={"question": question})
    assert res.status_code == 200, res.text
    return res.json()


def test_sample_documents_are_loaded(client):
    names = [d["filename"] for d in client.get("/api/documents").json()]
    assert names == ["expense-policy.md", "on-call-guide.md", "remote-work-policy.md"]


def test_answers_cite_the_right_document(client):
    data = ask(client, "How many days a week can I work remotely?")
    assert data["grounded"] and data["mode"] == "extractive"
    assert data["sources"][0]["document"] == "remote-work-policy.md"
    assert "three days a week" in data["answer"] and "[1]" in data["answer"]


def test_refuses_when_the_documents_do_not_cover_the_question(client):
    data = ask(client, "What is the capital of Australia?")
    assert data == {"answer": NOT_FOUND, "grounded": False, "mode": "none", "sources": []}


def test_upload_pdf_then_ask_then_delete(client):
    pdf = (FIXTURES / "security-handbook.pdf").read_bytes()
    res = client.post("/api/documents", files={"file": ("security-handbook.pdf", pdf, "application/pdf")})
    assert res.status_code == 201 and res.json()["chunks"] >= 1
    data = ask(client, "How long must passwords be?")
    assert data["sources"][0]["document"] == "security-handbook.pdf" and "14 characters" in data["answer"]
    assert client.delete(f"/api/documents/{res.json()['id']}").status_code == 204
    assert not ask(client, "How long must passwords be?")["grounded"]
    assert client.delete(f"/api/documents/{res.json()['id']}").status_code == 404


def test_rejects_bad_uploads_and_questions(client):
    assert (
        client.post("/api/documents", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 415
    )
    big = b"a " * (1024 * 1024 + 10)
    assert client.post("/api/documents", files={"file": ("big.txt", big, "text/plain")}).status_code == 413
    assert client.post("/api/documents", files={"file": ("empty.txt", b"   ", "text/plain")}).status_code == 422
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 422


def test_home_page_is_served(client):
    res = client.get("/")
    assert res.status_code == 200 and "Document Assistant" in res.text


def _claude(handler):
    return ClaudeAnswerer(
        "test-key",
        "test-model",
        httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.anthropic.com"),
    )


def test_claude_answerer_sends_sources_and_returns_text():
    seen = {}

    def handler(request: httpx.Request):
        seen["headers"], seen["body"] = request.headers, request.read().decode()
        return httpx.Response(200, json={"content": [{"type": "text", "text": "Up to three days [1]."}]})

    client = TestClient(create_app(load_samples=True, answerer=_claude(handler)))
    data = ask(client, "How many days a week can I work remotely?")
    assert data == {**data, "answer": "Up to three days [1].", "mode": "llm", "grounded": True}
    assert seen["headers"]["x-api-key"] == "test-key"
    assert "[1] (remote-work-policy.md)" in seen["body"] and "test-model" in seen["body"]


def test_falls_back_to_extractive_answer_when_the_llm_fails():
    client = TestClient(create_app(load_samples=True, answerer=_claude(lambda r: httpx.Response(500))))
    data = ask(client, "How many days a week can I work remotely?")
    assert data["mode"] == "extractive" and "three days a week" in data["answer"]
