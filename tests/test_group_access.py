from src.retrieval.retriever import Retriever
from src.embeddings.voyage_embeddings import VoyageEmbedder
from src.vectorstores.pinecone_store import PineconeStore
from src.security.user_context import UserContext

def main():

    retriever = Retriever(
        embedder=VoyageEmbedder(),
        store=PineconeStore(),
    )

    platform_user = UserContext(
        user_id="eng-01",
        role="employee",
        departments=["engineering"],
        groups=["platform-team"],
    )

    normal_engineer = UserContext(
        user_id="eng-02",
        role="employee",
        departments=["engineering"],
        groups=[],
    )

    query = (
        "What is the platform team strategy?"
    )

    print("\nPLATFORM USER")

    results = retriever.retrieve(
        query=query,
        user_context=platform_user,
    )

    for r in results:
        print(r["source"])

    print("\nNORMAL ENGINEER")

    results = retriever.retrieve(
        query=query,
        user_context=normal_engineer,
    )

    for r in results:
        print(r["source"])

if __name__ == "__main__":
    main()