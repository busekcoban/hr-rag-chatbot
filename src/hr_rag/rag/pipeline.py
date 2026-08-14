class RagPipeline:
    def __init__(self, retriever, generator):
        self._retriever = retriever
        self._generator = generator

    def is_index_ready(self) -> bool:
        return self._retriever.count() > 0

    def answer(self, question: str) -> dict:
        chunks = self._retriever.retrieve(question)
        answer_text = self._generator.generate(question, chunks)
        return {
            "answer": answer_text,
        }