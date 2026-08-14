from dotenv import load_dotenv
load_dotenv()

import os
class Settings:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.generation_model = os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash")
        self.embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        self.persist_dir = os.getenv("CHROMA_PERSIST_DIR", ".chroma")
        self.docs_dir = os.getenv("DOCS_DIR", "data")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "120"))
        self.top_k = int(os.getenv("TOP_K", "5"))
    def require_api_key(self):
        if not self.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not defined in .env")
        return self.gemini_api_key

settings = Settings()