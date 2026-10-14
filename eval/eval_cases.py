from dataclasses import dataclass
from typing import Optional

@dataclass
class EvalCase:
    question: str
    expected_topic: str
    expected_answer: str
    metadata_filter: Optional[dict] = None
    expect_retrieval: bool = True

EVAL_CASES = [

    # --------------------------------------------------
    # Security Policy
    # --------------------------------------------------

    EvalCase(
        question="How do I reset my password?",
        expected_topic="password reset process",
        expected_answer=(
            "Use the self-service password reset portal."
        ),
    ),

    EvalCase(
        question="How long should passwords be?",
        expected_topic="password requirements",
        expected_answer=(
            "Passwords must be at least 14 characters."
        ),
    ),

    EvalCase(
        question="How often do passwords expire?",
        expected_topic="password expiration policy",
        expected_answer=(
            "Passwords expire every 90 days."
        ),
    ),

    EvalCase(
        question="What happens after five failed logins?",
        expected_topic="account lockout policy",
        expected_answer=(
            "Accounts are automatically locked."
        ),
    ),

    # --------------------------------------------------
    # Remote Work
    # --------------------------------------------------

    EvalCase(
        question="How many days per week can I work remotely?",
        expected_topic="remote work allowance",
        expected_answer=(
            "Up to 3 days per week with manager approval."
        ),
    ),

    EvalCase(
        question="Can I use my personal laptop for work?",
        expected_topic="personal device policy",
        expected_answer=(
            "Personal devices are not permitted."
        ),
    ),

    EvalCase(
        question="Do I need VPN access when working remotely?",
        expected_topic="vpn requirement",
        expected_answer=(
            "Remote work must use the company VPN."
        ),
    ),

    # --------------------------------------------------
    # Data Retention
    # --------------------------------------------------

    EvalCase(
        question="How long do we retain customer data?",
        expected_topic="data retention period",
        expected_answer=(
            "Minimum seven years."
        ),
    ),

    EvalCase(
        question="What happens to PII after a deletion request?",
        expected_topic="PII deletion policy",
        expected_answer=(
            "PII must be deleted within 30 days."
        ),
    ),

    # --------------------------------------------------
    # No-Answer Cases
    # --------------------------------------------------

    EvalCase(
        question="What is the company's maternity leave policy?",
        expected_topic="not found",
        expected_answer="The information is not available",
        expect_retrieval=False,
    ),

    EvalCase(
        question="What is the CEO's phone number?",
        expected_topic="not found",
        expected_answer="The information is not available",
        expect_retrieval=False,
    ),
]