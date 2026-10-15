from src.models.document import Document
from src.ingestion.chunker import TextChunker
from src.ingestion.ingest import DocumentIngestor
from src.embeddings.voyage_embeddings import VoyageEmbedder
from src.vectorstores.pinecone_store import PineconeStore

def main():

    docs = [

        Document(
            id="platform_strategy",
            text="""
            Platform team strategy includes
            migration to service mesh architecture.
            """,
            metadata={
                "source": "platform_strategy.pdf",
                "access_level": "public",
                "department": "engineering",
                "owner_group": "platform-team",
            },
        ),

        Document(
            id="executive_strategy",
            text="""
            Executive board growth strategy
            targets a 25 percent market expansion.
            """,
            metadata={
                "source": "executive_strategy.pdf",
                "access_level": "admin",
                "department": "leadership",
                "owner_group": "executive-team",
            },
        ),
    ]

    ingestor = DocumentIngestor(
        chunker=TextChunker(),
        embedder=VoyageEmbedder(),
        store=PineconeStore(),
    )

    count = ingestor.ingest_documents(docs)

    print(
        f"Ingested {count} chunks"
    )

if __name__ == "__main__":
    main()