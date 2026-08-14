from rank_bm25 import BM25Okapi


class HybridRetriever:
    def __init__(self, vector_store, embedder, all_chunks: list[dict]):
        self._vector_store = vector_store
        self._embedder = embedder
        self._chunks = all_chunks
        tokenized = [c["text"].lower().split() for c in all_chunks]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        query_embedding = self._embedder.embed_query(question)
        dense_results = self._vector_store.query(query_embedding, top_k=top_k * 2)

        tokenized_query = question.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)

        combined = {}
        for r in dense_results:
            combined[r["id"]] = combined.get(r["id"], 0) + (1 - r["distance"])

        for chunk, score in zip(self._chunks, bm25_scores):
            if score > 0:
                normalized = score / (max(bm25_scores) or 1)
                combined[chunk["id"]] = combined.get(chunk["id"], 0) + normalized

        ranked_ids = sorted(combined, key=combined.get, reverse=True)[:top_k]
        chunk_by_id = {c["id"]: c for c in self._chunks}
        return [chunk_by_id[i] for i in ranked_ids]