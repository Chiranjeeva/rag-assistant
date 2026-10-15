from src.retrieval.retriever import Retriever
from src.generation.claude_generator import ClaudeGenerator
from src.security.user_context import (
    UserContext
)
from datetime import datetime
from src.audit.query_audit import (
    QueryAudit
)
from src.audit.audit_logger import (
    AuditLogger
)

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
        user_context: UserContext | None = None,    
        top_k: int = 5,
        namespace: str = "default",
        metadata_filter: dict | None = None,
    ) -> str:

        chunks = self.retriever.retrieve(
            query=question,
            user_context=user_context,
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
        user_context: UserContext | None = None,
        top_k: int = 5,
        namespace: str = "default",
        metadata_filter: dict | None = None,
    ) -> dict:

        chunks = self.retriever.retrieve(
            query=question,
            user_context=user_context,
            top_k=top_k,
            namespace=namespace,
            metadata_filter=metadata_filter,
        )

        if user_context:
            audit_record = QueryAudit(
                timestamp=datetime.utcnow().isoformat(),
                user_id=user_context.user_id,
                role=user_context.role,
                question=question,
                retrieved_sources=[
                    chunk["source"]
                    for chunk in chunks
                ],
                retrieval_scores=[
                    chunk["score"]
                    for chunk in chunks
                ],
            )

            AuditLogger.log(
                audit_record
            )

        answer = self.generator.generate(
            question=question,
            retrieved_chunks=chunks,
        )

        return {
            "answer": answer,
            "sources": chunks,
        }