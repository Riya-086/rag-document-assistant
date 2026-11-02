import io

from pypdf import PdfReader

ALLOWED = (".md", ".txt", ".pdf")


class UnsupportedFile(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:  # pypdf raises several error types for damaged files
            raise UnsupportedFile("That PDF could not be read.") from exc
    if name.endswith((".md", ".txt")):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UnsupportedFile("Text files must be UTF-8.") from exc
    raise UnsupportedFile(f"Upload a {', '.join(ALLOWED)} file.")
