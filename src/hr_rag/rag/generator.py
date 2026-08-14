from google import genai
from google.genai import types


SYSTEM_INSTRUCTION = (
    "Act like a HR Policy Assistant. Answer only based on the document excerpts provided to you. "
    "If the answer is not in the documents, say so clearly instead of guessing."
)

def build_prompt(question: str, chunks: list[dict]) -> str:
    context_parts = []
    for c in chunks:
        context_parts.append(f"Source: {c['source']}\n{c['text']}")
    context = "\n\n-\n\n".join(context_parts)
    return f"Documents:\n{context}\n\nQuestion: {question}\n\nAnswer:"

class GeminiGenerator:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, question: str, chunks: list[dict]) -> str:
        prompt = build_prompt(question, chunks)
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2,
            ),
        )
        return response.text