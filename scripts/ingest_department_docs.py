from src.models.document import Document
from src.ingestion.chunker import TextChunker
from src.ingestion.ingest import DocumentIngestor
from src.embeddings.voyage_embeddings import VoyageEmbedder
from src.vectorstores.pinecone_store import PineconeStore

def main():

    documents = [

        Document(
            id="engineering_architecture",
            text="""
            The 2027 software architecture
            migration will move all services
            to Kubernetes.
            """,
            metadata={
                "source": "engineering_architecture.pdf",
                "access_level": "public",
                "department": "engineering",
            }
        ),

        Document(
            id="finance_forecast",
            text="""
            Revenue forecast for 2027 is
            projected to exceed targets by 12%.
            """,
            metadata={
                "source": "finance_forecast.pdf",
                "access_level": "public",
                "department": "finance",
            }
        ),
    ]

    ingestor = DocumentIngestor(
        chunker=TextChunker(),
        embedder=VoyageEmbedder(),
        store=PineconeStore(),
    )

    count = ingestor.ingest_documents(documents)

    print(f"Ingested {count} chunks")

if __name__ == "__main__":
    main()