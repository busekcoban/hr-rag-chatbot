import streamlit as st

from hr_rag.config import settings
from hr_rag.ingestion.document_loader import load_all_documents
from hr_rag.rag.chunking import chunk_documents
from hr_rag.rag.embeddings import GeminiEmbedder
from hr_rag.rag.generator import GeminiGenerator
from hr_rag.rag.pipeline import RagPipeline
from hr_rag.rag.retriever import HybridRetriever
from hr_rag.rag.vector_store import ChromaVectorStore

st.set_page_config(page_title="HR Policy Assistant", page_icon="📋")

@st.cache_resource(show_spinner=False)
def get_pipeline():
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
    return RagPipeline(retriever, generator)

st.title("📋 HR Policy Assistant")

pipeline = get_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Please ask your question")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking.."):
            result = pipeline.answer(question)
        st.markdown(result["answer"])

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

