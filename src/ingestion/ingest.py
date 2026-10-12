from src.ingestion.chunker import TextChunker
from src.embeddings.voyage_embeddings import VoyageEmbedder
from src.vectorstores.pinecone_store import PineconeStore
from src.models.document import Document
import hashlib

class DocumentIngestor:
    """
    Responsible for:

    1. Chunking documents
    2. Generating embeddings
    3. Upserting vectors into Pinecone
    """

    def __init__(
        self,
        chunker: TextChunker,
        embedder: VoyageEmbedder,
        store: PineconeStore,
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.store = store

    def ingest_documents(
        self,
        documents: list[Document],
        namespace: str = "default",
    ) -> int:

        vectors = []

        for document in documents:

            chunks = self.chunker.chunk(
                document.text
            )

            texts = [
                chunk["text"]
                for chunk in chunks
            ]

            embeddings = self.embedder.embed_batch(
                texts
            )

            for chunk, embedding in zip(
                chunks,
                embeddings
            ):

                vector_id = (
                    f"{document.id}_chunk_"
                    f"{chunk['chunk_index']}"
                )

                metadata = {
                    **document.metadata,
                    "text": chunk["text"],
                    "source_id": document.id,
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": chunk["total_chunks"],
                }

                vectors.append(
                    (
                        vector_id,
                        embedding,
                        metadata,
                    )
                )

        if vectors:

            self.store.upsert_vectors(
                vectors=vectors,
                namespace=namespace,
            )

        return len(vectors)

    @property
    def content_hash(self):

        return hashlib.sha256(
            self.text.encode("utf-8")
        ).hexdigest()