from google import genai


class GeminiEmbedder:
    def __init__(self,api_key:str,model = "gemini-embedding-001"):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def embed_documents(self,texts: list[str]) -> list[list[float]]:
        response = self._client.models.embed_content(
            model=self._model,
            contents=texts,
        )
        return [e.values for e in response.embeddings]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
