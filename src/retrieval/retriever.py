from src.embeddings.voyage_embeddings import VoyageEmbedder
from src.vectorstores.pinecone_store import PineconeStore
from src.config.settings import (
    DEFAULT_TOP_K,
    SIMILARITY_THRESHOLD,
)

class Retriever:
    """
    Handles semantic retrieval from Pinecone.
    """

    def __init__(
        self,
        embedder: VoyageEmbedder,
        store: PineconeStore,
    ):
        self.embedder = embedder
        self.store = store

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        namespace: str = "default",
        metadata_filter: dict | None = None,
    ) -> list[dict]:

        query_embedding = self.embedder.embed_text(
            query
        )

        results = self.store.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            metadata_filter=metadata_filter,
        )

        matches = []

        for match in results.matches:

            if match.score < SIMILARITY_THRESHOLD:
                continue

            matches.append(
                {
                    "id": match.id,
                    "score": match.score,
                    "text": match.metadata.get(
                        "text",
                        ""
                    ),
                    "source": match.metadata.get(
                        "source",
                        "unknown"
                    ),
                    "chunk_index": match.metadata.get(
                        "chunk_index",
                        0
                    ),
                    "total_chunks": match.metadata.get(
                        "total_chunks",
                        0
                    ),
                    "metadata": match.metadata,
                }
            )

        return matches