from pathlib import Path
from pypdf import PdfReader

def load_file(file_path: Path) -> str:
    reader = PdfReader(file_path)
    text_parts = []
    for p in reader.pages:
        text_parts.append(p.extract_text())
    return "\n".join(text_parts)

def load_all_documents(docs_dir: str) -> list[dict]:
    documents = []
    for pdf_path in Path(docs_dir).glob("*.pdf"):
        text = load_file(pdf_path)
        documents.append({"source": pdf_path.name, "text": text})
    return documents