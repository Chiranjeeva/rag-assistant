from src.embeddings.voyage_embeddings import (
    VoyageEmbedder
)

from src.vectorstores.pinecone_store import (
    PineconeStore
)

from src.retrieval.retriever import (
    Retriever
)

from src.security.user_context import (
    UserContext
)

def main():

    retriever = Retriever(
        embedder=VoyageEmbedder(),
        store=PineconeStore(),
    )

    employee = UserContext(
        user_id="emp-001",
        role="employee",
    )

    hr = UserContext(
        user_id="hr-001",
        role="hr",
    )

    question = (
        "What is the salary budget for 2027?"
    )

    print("\nEMPLOYEE RESULTS\n")

    employee_results = retriever.retrieve(
        query=question,
        user_context=employee,
    )

    print(
        f"Retrieved: {len(employee_results)} chunks"
    )

    print("\nHR RESULTS\n")

    hr_results = retriever.retrieve(
        query=question,
        user_context=hr,
    )

    print(
        f"Retrieved: {len(hr_results)} chunks"
    )

    for result in hr_results:

        print(
            f"Source: {result['source']}"
        )

if __name__ == "__main__":
    main()