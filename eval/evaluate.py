import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hr_rag.config import settings
from hr_rag.ingestion.document_loader import load_all_documents
from hr_rag.rag.chunking import chunk_documents
from hr_rag.rag.embeddings import GeminiEmbedder
from hr_rag.rag.generator import GeminiGenerator
from hr_rag.rag.pipeline import RagPipeline
from hr_rag.rag.retriever import HybridRetriever
from hr_rag.rag.vector_store import ChromaVectorStore


def load_golden_set(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def evaluate_one(pipeline, item: dict) -> bool:
    result = pipeline.answer(item["question"])
    answer_lower = result["answer"].lower()
    keywords_found = all(kw.lower() in answer_lower for kw in item["expected_keywords"])
    return keywords_found


def is_chunk_relevant(chunk: dict, expected_keywords: list[str]) -> bool:
    text_lower = chunk["text"].lower()
    return any(kw.lower() in text_lower for kw in expected_keywords)


def retrieval_metrics(retriever, item: dict, k: int = 5) -> dict:
    retrieved = retriever.retrieve(item["question"], top_k=k)
    relevance = [is_chunk_relevant(c, item["expected_keywords"]) for c in retrieved]

    num_relevant_retrieved = sum(relevance)
    precision = num_relevant_retrieved / k
    recall = 1.0 if num_relevant_retrieved > 0 else 0.0

    rr = 0.0
    for rank, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            rr = 1 / rank
            break

    return {"precision": precision, "recall": recall, "rr": rr}


def main() -> int:
    api_key = settings.require_api_key()
    documents = load_all_documents(settings.docs_dir)
    all_chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)

    embedder = GeminiEmbedder(api_key, settings.embedding_model)
    store = ChromaVectorStore(settings.persist_dir)
    if store.count() == 0 and all_chunks:
        embeddings = embedder.embed_documents([c["text"] for c in all_chunks])
        store.add_chunks(all_chunks, embeddings)

    retriever = HybridRetriever(store, embedder, all_chunks)
    generator = GeminiGenerator(api_key, settings.generation_model)
    pipeline = RagPipeline(retriever, generator)

    golden_set = load_golden_set("eval/golden_qa.jsonl")
    passed = 0
    all_metrics = []
    for item in golden_set:
        ok = evaluate_one(pipeline, item)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {item['question']}")
        if ok:
            passed += 1

        m = retrieval_metrics(retriever, item, k=settings.top_k)
        all_metrics.append(m)

    score = passed / len(golden_set)
    avg_precision = sum(m["precision"] for m in all_metrics) / len(all_metrics)
    avg_recall = sum(m["recall"] for m in all_metrics) / len(all_metrics)
    mrr = sum(m["rr"] for m in all_metrics) / len(all_metrics)

    print(f"\nAnswer score: {passed}/{len(golden_set)} ({score:.0%})")
    print(f"Precision@{settings.top_k}: {avg_precision:.2f}")
    print(f"Recall@{settings.top_k}: {avg_recall:.2f}")
    print(f"MRR: {mrr:.2f}")

    return 0 if score >= 0.7 else 1


if __name__ == "__main__":
    sys.exit(main())