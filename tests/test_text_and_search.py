from rag_assistant.search import BM25Index, Chunk
from rag_assistant.text import chunk, tokens


def test_tokens_drop_stopwords_and_normalise_plurals():
    assert tokens("What are the expenses for meals?") == ["expens", "meal"]


def test_chunks_respect_the_size_limit_and_keep_every_sentence():
    text = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(60))
    chunks = chunk(text, max_chars=200)
    assert len(chunks) > 5 and all(len(c) <= 200 for c in chunks)
    joined = " ".join(chunks)
    assert all(f"number {i} " in joined for i in range(60))
    assert chunks[0].split(". ")[-1] in chunks[1]  # one sentence of overlap


def test_bm25_ranks_the_relevant_chunk_first_and_ignores_unrelated_queries():
    index = BM25Index()
    index.add(Chunk(1, 1, "a.md", "Meals while travelling are reimbursed up to 60 dollars per day."))
    index.add(Chunk(2, 2, "b.md", "Engineers join the on-call rotation after three months."))
    index.add(Chunk(3, 3, "c.md", "The office has standing desks and a quiet room."))
    hits = index.search("How much is reimbursed for meals?")
    assert hits[0].chunk.id == 1
    assert index.search("quantum chromodynamics lecture") == []
    index.remove_document(1)
    assert all(h.chunk.id != 1 for h in index.search("meals reimbursed"))
