from src.embeddings.voyage_embeddings import VoyageEmbedder
from src.vectorstores.pinecone_store import PineconeStore
from src.config.settings import (
    DEFAULT_TOP_K,
    SIMILARITY_THRESHOLD,
)

from src.security.access_control import AccessControl
from src.security.user_context import UserContext

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
        user_context: UserContext | None = None,
        top_k: int = DEFAULT_TOP_K,
        namespace: str = "default",
        metadata_filter: dict | None = None,
    ) -> list[dict]:

        query_embedding = self.embedder.embed_text(
            query
        )

        # --------------------------------------------------
        # Permission-aware filtering
        # --------------------------------------------------

        if user_context:

            allowed_levels = (
                AccessControl.get_allowed_levels(
                    user_context.role
                )
            )

            metadata_filter = {
                "$and": [

                    {
                        "access_level": {
                            "$in": allowed_levels
                        }
                    },

                    {
                        "department": {
                            "$in": (
                                user_context.departments
                                + ["general"]
                            )
                        }
                    },

                    {
                        "owner_group": {
                            "$in": (
                                user_context.groups
                                + ["general"]
                            )
                        }
                    }
                ]
            }

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