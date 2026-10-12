from src.embeddings.voyage_embeddings import (
    VoyageEmbedder
)

from src.vectorstores.pinecone_store import (
    PineconeStore
)

from src.retrieval.retriever import (
    Retriever
)

def main():

    embedder = VoyageEmbedder()

    store = PineconeStore()

    retriever = Retriever(
        embedder=embedder,
        store=store,
    )

    results = retriever.retrieve(
        "How do I reset my password?"
    )

    print()

    print(
        f"Retrieved {len(results)} chunks"
    )

    print()

    for result in results:

        print(
            f"Score: {result['score']:.3f}"
        )

        print(
            f"Source: {result['source']}"
        )

        print()

        print(
            result['text']
        )

        print("-" * 60)


if __name__ == "__main__":
    main()