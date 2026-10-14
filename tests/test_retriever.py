from src.embeddings.voyage_embeddings import VoyageEmbedder
from src.vectorstores.pinecone_store import PineconeStore
from src.retrieval.retriever import Retriever

def main():

    retriever = Retriever(
        embedder=VoyageEmbedder(),
        store=PineconeStore(),
    )

    results = retriever.retrieve(
        "Can I use my personal laptop for work?"
    )

    print(f"\nRetrieved {len(results)} chunks\n")

    for i, result in enumerate(results, start=1):

        print("=" * 80)
        print(f"Chunk {i}")
        print(f"Score: {result['score']:.3f}")
        print(f"Source: {result['source']}")
        print()
        print(result["text"])
        print()

if __name__ == "__main__":
    main()