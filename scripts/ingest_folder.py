from pathlib import Path

from src.loaders.pdf_loader import PDFLoader
from src.ingestion.chunker import TextChunker
from src.ingestion.ingest import DocumentIngestor
from src.embeddings.voyage_embeddings import VoyageEmbedder
from src.vectorstores.pinecone_store import PineconeStore


def main():

    pdf_directory = Path("data/raw")

    pdf_files = list(pdf_directory.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data/raw")
        return

    loader = PDFLoader()

    chunker = TextChunker()

    embedder = VoyageEmbedder()

    store = PineconeStore()

    ingestor = DocumentIngestor(
        chunker=chunker,
        embedder=embedder,
        store=store,
    )

    total_chunks = 0

    for pdf_path in pdf_files:

        print(f"\nProcessing {pdf_path.name}")

        document = loader.load(str(pdf_path))

        chunk_count = ingestor.ingest_documents(
            documents=[document]
        )

        total_chunks += chunk_count

        print(
            f"Ingested {chunk_count} chunks"
        )

    print("\n" + "=" * 50)

    print(
        f"Finished. Total chunks ingested: {total_chunks}"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()