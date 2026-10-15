from src.models.document import Document
from src.ingestion.chunker import TextChunker
from src.ingestion.ingest import DocumentIngestor
from src.embeddings.voyage_embeddings import VoyageEmbedder
from src.vectorstores.pinecone_store import PineconeStore

def main():

    documents = [

        Document(
            id="employee_handbook",
            text="""
            Employees receive 20 days of annual leave.

            Company holidays are published annually.

            Employees are expected to follow all
            company policies and procedures.
            """,
            metadata={
                "source": "employee_handbook.pdf",
                "access_level": "public",
                "department": "general"
            }
        ),

        Document(
            id="salary_plan_2027",
            text="""
            Salary budgets for 2027 include
            an average salary increase of 8%.

            Bonus allocations will be determined
            by department leadership.

            This document is restricted
            to HR personnel only.
            """,
            metadata={
                "source": "salary_plan_2027.pdf",
                "access_level": "hr",
                "department": "hr"
            }
        )

    ]

    ingestor = DocumentIngestor(
        chunker=TextChunker(),
        embedder=VoyageEmbedder(),
        store=PineconeStore(),
    )

    count = ingestor.ingest_documents(
        documents=documents
    )

    print(
        f"Ingested {count} chunks"
    )

if __name__ == "__main__":
    main()