from src.embeddings.voyage_embeddings import VoyageEmbedder
from src.vectorstores.pinecone_store import PineconeStore
from src.retrieval.retriever import Retriever
from src.security.user_context import UserContext


def main():

    retriever = Retriever(
        embedder=VoyageEmbedder(),
        store=PineconeStore(),
    )

    engineering_user = UserContext(
        user_id="eng-001",
        role="employee",
        departments=["engineering"]
    )

    finance_user = UserContext(
        user_id="fin-001",
        role="employee",
        departments=["finance"]
    )

    question = (
        "What is the 2027 software architecture migration?"
    )

    eng_results = retriever.retrieve(
        query=question,
        user_context=engineering_user,
    )

    finance_results = retriever.retrieve(
        query=question,
        user_context=finance_user,
    )

    print("\nENGINEERING")

    for result in eng_results:
        print(result["source"])

    print("\nFINANCE")

    for result in finance_results:
        print(result["source"])


if __name__ == "__main__":
    main()