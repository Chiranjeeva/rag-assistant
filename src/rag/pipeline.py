from src.retrieval.retriever import Retriever
from src.generation.claude_generator import ClaudeGenerator

class RAGPipeline:
    """
    End-to-end Retrieval Augmented Generation pipeline.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: ClaudeGenerator,
    ):
        self.retriever = retriever
        self.generator = generator

    def query(
        self,
        question: str,
        top_k: int = 5,
        namespace: str = "default",
        metadata_filter: dict | None = None,
    ) -> str:

        chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            namespace=namespace,
            metadata_filter=metadata_filter,
        )

        answer = self.generator.generate(
            question=question,
            retrieved_chunks=chunks,
        )

        return answer

    def query_with_sources(
        self,
        question: str,
        top_k: int = 5,
        namespace: str = "default",
        metadata_filter: dict | None = None,
    ) -> dict:

        chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            namespace=namespace,
            metadata_filter=metadata_filter,
        )

        answer = self.generator.generate(
            question=question,
            retrieved_chunks=chunks,
        )

        return {
            "answer": answer,
            "sources": chunks,
        }