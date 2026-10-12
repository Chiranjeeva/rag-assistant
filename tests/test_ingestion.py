from src.models.document import Document

from src.ingestion.chunker import (
    TextChunker
)

from src.ingestion.ingest import (
    DocumentIngestor
)

from src.embeddings.voyage_embeddings import (
    VoyageEmbedder
)

from src.vectorstores.pinecone_store import (
    PineconeStore
)

def main():

    document = Document(
        id="policy-001",
        text="""
        Password reset requests must use the
        company self-service portal.

        Passwords require at least
        fourteen characters.

        Contact IT support if locked out.
        """,
        metadata={
            "source": "security_policy.pdf"
        }
    )

    chunker = TextChunker(
        chunk_size=20,
        chunk_overlap=5,
    )

    embedder = VoyageEmbedder()

    store = PineconeStore()

    ingestor = DocumentIngestor(
        chunker=chunker,
        embedder=embedder,
        store=store,
    )

    total_vectors = ingestor.ingest_documents(
        documents=[document]
    )

    print(
        f"Ingested {total_vectors} chunks"
    )

if __name__ == "__main__":
    main()