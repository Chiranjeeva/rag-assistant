from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class QueryAudit:

    timestamp: str
    user_id: str
    role: str
    question: str
    retrieved_sources: list[str] = field(
        default_factory=list
    )
    retrieval_scores: list[float] = field(
        default_factory=list
    )
