import json
from pathlib import Path

from src.audit.query_audit import (
    QueryAudit
)

class AuditLogger:

    LOG_FILE = Path(
        "audit_logs.jsonl"
    )

    @classmethod
    def log(
        cls,
        audit_record: QueryAudit
    ):

        with open(
            cls.LOG_FILE,
            "a"
        ) as f:

            f.write(
                json.dumps(
                    audit_record.__dict__
                )
            )

            f.write("\n")