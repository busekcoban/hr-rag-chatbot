def chunk_text(text:str,chunk_size: int=800, chunk_overlap: int =120) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start = end - chunk_overlap
    return chunks


def chunk_documents(documents: list[dict],chunk_size : int = 800, chunk_overlap: int = 120) -> list[dict]:
    result = []
    for doc in documents:
        pieces = chunk_text(doc["text"], chunk_size, chunk_overlap)
        for i, piece in enumerate(pieces):
            result.append({
                "id": f"{doc['source']}::{i}",
                "text": piece,
                "source": doc["source"],
            })
    return result



