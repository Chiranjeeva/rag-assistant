from src.embeddings.voyage_embeddings import VoyageEmbedder
from src.vectorstores.pinecone_store import PineconeStore
from src.retrieval.retriever import Retriever
from src.generation.claude_generator import ClaudeGenerator
from src.rag.pipeline import RAGPipeline

def main():

    embedder = VoyageEmbedder()
    store = PineconeStore()
    retriever = Retriever(
        embedder=embedder,
        store=store,
    )
    generator = ClaudeGenerator()
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    result = pipeline.query(
        "How do I reset my password?"
    )

    print()
    print(result)
    print()


if __name__ == "__main__":
    main()