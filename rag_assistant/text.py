import re

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be but by can could do does for from how i if in into is it its may might must of on or our "
    "shall should so that the their there these this to was we were what when where which who why will with would "
    "you your".split()
)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(" ".join(text.split())) if s.strip()]


def tokens(text: str) -> list[str]:
    """Lower-case word tokens without stop words, with a light plural/verb-suffix trim."""
    out = []
    for word in _WORD.findall(text.lower()):
        if word in STOPWORDS:
            continue
        for suffix in ("ing", "es", "s"):
            if len(word) > len(suffix) + 3 and word.endswith(suffix):
                word = word[: -len(suffix)]
                break
        out.append(word)
    return out


def chunk(text: str, max_chars: int = 700) -> list[str]:
    """Split into chunks of whole sentences, keeping paragraphs together where possible.

    Neighbouring chunks share one sentence of overlap so an answer that spans a
    boundary is still retrievable.
    """
    chunks: list[str] = []
    current: list[str] = []
    # Markdown headings become their own sentence so they never merge into the next one.
    text = re.sub(r"^#{1,6}\s*(.+?)\s*$", r"\1.", text, flags=re.MULTILINE)
    for paragraph in re.split(r"\n\s*\n", text):
        for sentence in sentences(paragraph):
            if current and len(" ".join(current + [sentence])) > max_chars:
                chunks.append(" ".join(current))
                current = current[-1:] if len(current[-1]) < max_chars // 2 else []
            current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks
