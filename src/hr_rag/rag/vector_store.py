import chromadb

# dense(embedding) search part.

class ChromaVectorStore:
    def __init__(self, persist_dir: str, collection_name: str = "hr_policies"):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def count(self) -> int:
        return self._collection.count()

    def add_chunks(self, chunk_dicts: list[dict], embeddings: list[list[float]]) -> None:
        if len(chunk_dicts) != len(embeddings):
            raise ValueError("chunk count should be equal to embedding count")
        if not chunk_dicts:
            return
        self._collection.add(
            ids=[c["id"] for c in chunk_dicts],
            embeddings=embeddings,
            documents=[c["text"] for c in chunk_dicts],
            metadatas=[{"source": c["source"]} for c in chunk_dicts],
        )

    def query(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        if self.count() == 0:
            return []
        top_k = min(top_k, self.count())
        raw = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)
        results = []
        for i in range(len(raw["ids"][0])):
            results.append({
                "id": raw["ids"][0][i],
                "text": raw["documents"][0][i],
                "source": raw["metadatas"][0][i]["source"],
                "distance": raw["distances"][0][i],
            })
        return results