import voyageai

from src.config.settings import (
    VOYAGE_API_KEY,
    EMBEDDING_MODEL,
)

class VoyageEmbedder:
    """
    Handles embedding operations using Voyage AI.
    """

    def __init__(self):
        self.client = voyageai.Client(
            api_key=VOYAGE_API_KEY
        )

    def embed_text(self, text: str) -> list:
        """
        Generate a single embedding vector.
        """

        response = self.client.embed(
            texts=[text],
            model=EMBEDDING_MODEL
        )

        return response.embeddings[0]

    def embed_batch(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """
        Generate embedding vectors for multiple texts.
        """

        response = self.client.embed(
            texts=texts,
            model=EMBEDDING_MODEL
        )

        return response.embeddings