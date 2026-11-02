"""Turns retrieved chunks into an answer: extractively by default, or with Claude when a key is set."""

import logging

import httpx

from .search import Hit
from .text import sentences, tokens

logger = logging.getLogger(__name__)
NOT_FOUND = "I couldn't find that in the documents."
SYSTEM_PROMPT = (
    "You answer questions using only the numbered sources provided. Cite every claim with its source "
    "number in square brackets, like [1]. Be concise. If the sources do not contain the answer, reply "
    f"exactly: {NOT_FOUND}"
)


def extractive_answer(question: str, hits: list[Hit], max_sentences: int = 3) -> str:
    """Picks the sentences that share the most terms with the question, with citations."""
    wanted = set(tokens(question))
    scored = []
    for n, hit in enumerate(hits, start=1):
        for position, sentence in enumerate(sentences(hit.chunk.text)):
            overlap = len(wanted & set(tokens(sentence)))
            if overlap and len(sentence.split()) >= 5:  # skip headings and fragments
                scored.append((overlap, -n, -position, sentence, n))
    best = sorted(scored, reverse=True)[:max_sentences]
    best.sort(key=lambda s: (-s[1], -s[2]))  # restore reading order
    return " ".join(f"{sentence} [{n}]" for *_, sentence, n in best) or NOT_FOUND


class ClaudeAnswerer:
    def __init__(self, api_key: str, model: str, client: httpx.Client | None = None):
        self.api_key, self.model = api_key, model
        self.client = client or httpx.Client(base_url="https://api.anthropic.com", timeout=30)

    def __call__(self, question: str, hits: list[Hit]) -> str:
        sources = "\n\n".join(f"[{n}] ({h.chunk.document})\n{h.chunk.text}" for n, h in enumerate(hits, 1))
        response = self.client.post(
            "/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            json={
                "model": self.model,
                "max_tokens": 500,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"}],
            },
        )
        response.raise_for_status()
        text = "".join(b.get("text", "") for b in response.json().get("content", []) if b.get("type") == "text")
        return text.strip() or NOT_FOUND
