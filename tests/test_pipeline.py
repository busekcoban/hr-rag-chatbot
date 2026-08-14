from hr_rag.rag.pipeline import RagPipeline

class FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks
    def retrieve(self, question, top_k=5):
        return self._chunks
    def count(self):
        return len(self._chunks)

class FakeGenerator:
    def generate(self, question, chunks):
        return "Fake answer"

def test_pipeline_answer_returns_answer_and_sources():
    chunks = [{"id": "a::0", "text": "metin", "source": "a.pdf"}]
    pipeline = RagPipeline(retriever=FakeRetriever(chunks), generator=FakeGenerator())
    result = pipeline.answer("question")
    assert result["answer"] == "Fake answer"
    assert result["sources"] == chunks