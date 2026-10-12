from pinecone import (
    Pinecone,
    ServerlessSpec,
)

from src.config.settings import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_DIMENSIONS,
)

class PineconeStore:
    """
    Handles all Pinecone operations.
    """

    def __init__(self):

        self.pc = Pinecone(
            api_key=PINECONE_API_KEY
        )

        self.index_name = PINECONE_INDEX_NAME

        self._create_index_if_needed()

        self.index = self.pc.Index(
            self.index_name
        )

    def _create_index_if_needed(self):

        existing_indexes = [
            index.name
            for index in self.pc.list_indexes()
        ]

        if self.index_name not in existing_indexes:

            self.pc.create_index(
                name=self.index_name,
                dimension=EMBEDDING_DIMENSIONS,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION,
                )
            )

            print(
                f"Created index: {self.index_name}"
            )

        else:

            print(
                f"Using existing index: {self.index_name}"
            )

    def upsert_vectors(
        self,
        vectors,
        namespace="default",
        batch_size=100,
    ):

        for i in range(
            0,
            len(vectors),
            batch_size,
        ):

            batch = vectors[
                i:i + batch_size
            ]

            self.index.upsert(
                vectors=batch,
                namespace=namespace,
            )

    def query(
        self,
        vector,
        top_k=5,
        namespace="default",
        metadata_filter=None
    ):

        query_params = {
            "vector": vector,
            "top_k": top_k,
            "include_metadata": True,
            "namespace": namespace,
        }

        if metadata_filter:
            query_params["filter"] = metadata_filter

        return self.index.query(
            **query_params
        )

    def describe_index(self):

        return self.index.describe_index_stats()