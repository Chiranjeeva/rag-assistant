from src.embeddings.voyage_embeddings import VoyageEmbedder
from src.vectorstores.pinecone_store import PineconeStore
from src.retrieval.retriever import Retriever
from src.generation.claude_generator import ClaudeGenerator
from src.rag.pipeline import RAGPipeline

def create_rag_pipeline():

    embedder = VoyageEmbedder()
    store = PineconeStore()
    retriever = Retriever(
        embedder=embedder,
        store=store,
    )
    generator = ClaudeGenerator()

    return RAGPipeline(
        retriever=retriever,
        generator=generator,
    )