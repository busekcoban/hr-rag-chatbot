from hr_rag.rag.chunking import chunk_text, chunk_documents

def test_short_text_is_a_single_chunk():
    text = "This is a short text."
    chunks = chunk_text(text, chunk_size=800, chunk_overlap=120)
    assert len(chunks) == 1

def test_long_text_is_split_into_multiple_chunks():
    text = " ".join(["word"] * 2000)
    chunks = chunk_text(text, chunk_size=800, chunk_overlap=120)
    assert len(chunks) > 1

def test_chunk_documents_keeps_source_reference():
    documents = [{"source": "handbook.pdf", "text": "This is a test text."}]
    chunks = chunk_documents(documents, chunk_size=800, chunk_overlap=120)
    assert chunks[0]["source"] == "handbook.pdf"
    assert chunks[0]["id"] == "handbook.pdf::0"