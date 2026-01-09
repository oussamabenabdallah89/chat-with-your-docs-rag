from pypdf import PdfReader
from docx import Document

def parse_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join([(p.extract_text() or "") for p in reader.pages]).strip()

def parse_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs]).strip()

def parse_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()

def parse_file(path: str) -> str:
    p = path.lower()
    if p.endswith(".pdf"):
        return parse_pdf(path)
    if p.endswith(".docx"):
        return parse_docx(path)
    if p.endswith(".txt") or p.endswith(".md"):
        return parse_txt(path)
    raise ValueError("Unsupported file type.")
