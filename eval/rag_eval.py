"""
RAG Evaluation Harness

Evaluate whether your RAG pipeline is actually working.
Tests retrieval quality, answer faithfulness, and answer relevance
using an LLM-as-judge approach.

Works alongside rag_pipeline.py

Setup:
    pip install anthropic pinecone-client python-dotenv ragas

    .env file:
        ANTHROPIC_API_KEY=your-key-here
        PINECONE_API_KEY=your-key-here
        PINECONE_INDEX_NAME=your-index-name

Run:
    python rag_eval.py

Output:
    Prints a full evaluation report with per-question scores
    and aggregate metrics. Saves results to eval_results.json.
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Callable, Optional
from anthropic import Anthropic
from pinecone import Pinecone
import voyageai
from eval.eval_cases import EVAL_CASES
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

JUDGE_MODEL = "claude-sonnet-5-5"
EMBEDDING_MODEL = "voyage-4"
MIN_SIMILARITY = 0.5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalCase:
    """
    A single evaluation case from your golden dataset.

    question:        The question to ask the RAG system
    expected_topic:  What the answer should be about (used for retrieval eval)
    expected_answer: The correct answer (used for faithfulness eval)
    metadata_filter: Optional Pinecone filter (e.g. limit to a specific doc)
    """
    question: str
    expected_topic: str
    expected_answer: str
    metadata_filter: Optional[dict] = None


@dataclass
class RetrievalScore:
    hit: bool           # Did retrieval surface relevant content?
    top_chunk: str      # The highest-scoring chunk
    top_score: float    # Similarity score of the top chunk
    chunks_returned: int


@dataclass
class GenerationScore:
    faithfulness: float     # 0-1: Is the answer supported by context?
    relevance: float        # 0-1: Does the answer address the question?
    reasoning: str          # One-sentence explanation from the judge


@dataclass
class EvalResult:
    question: str
    expected_answer: str
    actual_answer: str
    retrieval: RetrievalScore
    generation: GenerationScore
    passed: bool            # True if both retrieval hit and gen scores above threshold


@dataclass
class EvalReport:
    timestamp: str
    total_cases: int
    passed: int
    failed: int
    retrieval_hit_rate: float
    avg_faithfulness: float
    avg_relevance: float
    results: list[EvalResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total_cases if self.total_cases > 0 else 0.0


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Adjust these based on your use case and risk tolerance.
# For regulated industries (legal, healthcare, finance) raise these significantly.
RETRIEVAL_HIT_THRESHOLD = 0.70   # Minimum retrieval hit rate to pass
FAITHFULNESS_THRESHOLD = 0.80    # Minimum faithfulness score to pass
RELEVANCE_THRESHOLD = 0.75       # Minimum relevance score to pass


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def embed_text(text: str) -> list[float]:
    response = vo.embed(
        model=EMBEDDING_MODEL,
        texts=[text]
    )
    return response.embeddings[0]

# ---------------------------------------------------------------------------
# Anthropic response helper
# ---------------------------------------------------------------------------

def extract_text_from_response(response) -> str:
    """
    Safely extract text from Anthropic responses.

    Handles:
    - TextBlock
    - ThinkingBlock
    - Future content block types
    """

    return "\n".join(
        getattr(block, "text", "")
        for block in response.content
    ).strip()

# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    question: str,
    expected_topic: str,
    retrieved_chunks: list[str],
    top_score: float,
) -> RetrievalScore:
    """
    Check whether retrieval surfaced relevant content.

    Uses LLM-as-judge for semantic matching rather than exact string
    comparison. This is more reliable for paraphrased or partial matches.

    If retrieved_chunks is empty, retrieval failed entirely.
    """
    if not retrieved_chunks:
        return RetrievalScore(
            hit=False,
            top_chunk="",
            top_score=0.0,
            chunks_returned=0,
        )

    chunks_text = "\n---\n".join(retrieved_chunks[:3])

    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=50,
        messages=[{
            "role": "user",
            "content": f"""Does the retrieved content contain information relevant to answering this question?

Question: {question}
Expected topic: {expected_topic}

Retrieved content:
{chunks_text}

Answer YES or NO only."""
        }]
    )

    judge_response = extract_text_from_response(
        response
    )

    hit = "YES" in judge_response.upper()

    return RetrievalScore(
        hit=hit,
        top_chunk=retrieved_chunks[0] if retrieved_chunks else "",
        top_score=round(top_score, 4),
        chunks_returned=len(retrieved_chunks),
    )


# ---------------------------------------------------------------------------
# Generation evaluation (LLM-as-judge)
# ---------------------------------------------------------------------------

def evaluate_generation(
    question: str,
    answer: str,
    context_chunks: list[str],
) -> GenerationScore:
    """
    Evaluate answer quality using an LLM as the judge.

    Faithfulness: Is every claim in the answer actually supported by the context?
    Relevance: Does the answer address what the question was asking?

    LLM-as-judge is not perfect but is significantly more scalable than
    human review and correlates well with human judgement at this task.
    """
    if not answer or not context_chunks:
        return GenerationScore(
            faithfulness=0.0,
            relevance=0.0,
            reasoning="No answer or context to evaluate.",
        )

    context_text = "\n\n".join(context_chunks)

    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"""You are evaluating the quality of an AI-generated answer.

Context provided to the AI:
{context_text}

Question asked: {question}
Answer generated: {answer}

Score the answer on two dimensions:

1. FAITHFULNESS (0.0 to 1.0): Is every factual claim in the answer directly supported by the context?
   - 1.0 = All claims are supported by the context
   - 0.5 = Some claims are supported, others are not verifiable from context
   - 0.0 = Answer contradicts or ignores the context entirely

2. RELEVANCE (0.0 to 1.0): Does the answer actually address what was asked?
   - 1.0 = Directly and completely answers the question
   - 0.5 = Partially answers but misses key aspects
   - 0.0 = Does not address the question at all

Respond in exactly this format:
FAITHFULNESS: [score]
RELEVANCE: [score]
REASONING: [one sentence]"""
        }]
    )

    text_parts = []

    for block in response.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)

    text = "\n".join(text_parts).strip()

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    try:
        faithfulness = float(lines[0].split(": ")[1])
        relevance = float(lines[1].split(": ")[1])
        reasoning = lines[2].split(": ", 1)[1] if len(lines) > 2 else "No reasoning provided."
    except (IndexError, ValueError) as e:
        logger.warning(f"Failed to parse judge response: {e}\nRaw: {text}")
        faithfulness = 0.5
        relevance = 0.5
        reasoning = "Could not parse judge response."

    return GenerationScore(
        faithfulness=round(faithfulness, 3),
        relevance=round(relevance, 3),
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Core eval runner
# ---------------------------------------------------------------------------

def run_eval(
    eval_cases: list[EvalCase],
    index: object,
    top_k: int = 5,
    namespace: str = "default",
    output_file: str = "eval_results.json",
) -> EvalReport:
    """
    Run the full evaluation suite against your RAG pipeline.

    For each eval case:
    1. Retrieves chunks from Pinecone
    2. Generates an answer
    3. Scores retrieval and generation independently
    4. Determines pass/fail

    Saves detailed results to output_file for later review.
    """
    results: list[EvalResult] = []

    for i, case in enumerate(eval_cases):
        logger.info(f"Evaluating case {i + 1}/{len(eval_cases)}: {case.question[:60]}...")

        # Step 1: Retrieve
        query_embedding = embed_text(case.question)
        query_params = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True,
            "namespace": namespace,
        }
        if case.metadata_filter:
            query_params["filter"] = case.metadata_filter

        pinecone_results = index.query(**query_params)

        print(f"\nQUESTION: {case.question}")

        for match in pinecone_results.matches:

            print(
                f"Score={match.score:.3f}"
            )

            print(
                f"Source={match.metadata.get('source', 'unknown')}"
            )
            
            print()

        retrieved_chunks = [
            match.metadata.get("text", "")
            for match in pinecone_results.matches
        ]

        top_score = pinecone_results.matches[0].score if pinecone_results.matches else 0.0

        # Step 2: Generate answer
        if retrieved_chunks:
            context = "\n\n---\n\n".join(retrieved_chunks)
            gen_response = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=1024,
                system="Answer questions using only the provided context. If the answer is not in the context, say so.",
                messages=[{
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {case.question}"
                }]
            )
            actual_answer = ""

            for block in gen_response.content:
                if hasattr(block, "text"):
                    actual_answer += block.text

        else:
            actual_answer = "I could not find relevant information to answer this question."

        # Step 3: Evaluate retrieval
        retrieval_score = evaluate_retrieval(
            question=case.question,
            expected_topic=case.expected_topic,
            retrieved_chunks=retrieved_chunks,
            top_score=top_score,
        )

        # Step 4: Evaluate generation
        generation_score = evaluate_generation(
            question=case.question,
            answer=actual_answer,
            context_chunks=retrieved_chunks,
        )

        # Step 5: Determine pass/fail
        if case.expect_retrieval:

            passed = (
                retrieval_score.hit
                and generation_score.faithfulness >= FAITHFULNESS_THRESHOLD
                and generation_score.relevance >= RELEVANCE_THRESHOLD
            )

        else:

            passed = (
                generation_score.faithfulness >= FAITHFULNESS_THRESHOLD
                and generation_score.relevance >= RELEVANCE_THRESHOLD
            )

        results.append(EvalResult(
            question=case.question,
            expected_answer=case.expected_answer,
            actual_answer=actual_answer,
            retrieval=retrieval_score,
            generation=generation_score,
            passed=passed,
        )
        )

    # Aggregate metrics
    passed_count = sum(1 for r in results if r.passed)
    retrieval_hit_rate = sum(1 for r in results if r.retrieval.hit) / len(results)
    avg_faithfulness = sum(r.generation.faithfulness for r in results) / len(results)
    avg_relevance = sum(r.generation.relevance for r in results) / len(results)

    report = EvalReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_cases=len(results),
        passed=passed_count,
        failed=len(results) - passed_count,
        retrieval_hit_rate=round(retrieval_hit_rate, 3),
        avg_faithfulness=round(avg_faithfulness, 3),
        avg_relevance=round(avg_relevance, 3),
        results=results,
    )

    # Save to file
    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": report.timestamp,
                "summary": {
                    "total_cases": report.total_cases,
                    "passed": report.passed,
                    "failed": report.failed,
                    "pass_rate": round(report.pass_rate, 3),
                    "retrieval_hit_rate": report.retrieval_hit_rate,
                    "avg_faithfulness": report.avg_faithfulness,
                    "avg_relevance": report.avg_relevance,
                },
                "thresholds": {
                    "faithfulness": FAITHFULNESS_THRESHOLD,
                    "relevance": RELEVANCE_THRESHOLD,
                    "retrieval_hit_rate": RETRIEVAL_HIT_THRESHOLD,
                },
                "results": [
                    {
                        "question": r.question,
                        "expected": r.expected_answer,
                        "actual": r.actual_answer,
                        "passed": r.passed,
                        "retrieval": {
                            "hit": r.retrieval.hit,
                            "top_score": r.retrieval.top_score,
                            "chunks_returned": r.retrieval.chunks_returned,
                        },
                        "generation": {
                            "faithfulness": r.generation.faithfulness,
                            "relevance": r.generation.relevance,
                            "reasoning": r.generation.reasoning,
                        },
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )

    logger.info(f"Results saved to {output_file}")
    return report


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(report: EvalReport) -> None:
    """Print a readable summary of eval results to the terminal."""
    print("\n" + "=" * 65)
    print("RAG EVALUATION REPORT")
    print("=" * 65)
    print(f"Timestamp:          {report.timestamp}")
    print(f"Total cases:        {report.total_cases}")
    print(f"Passed:             {report.passed} ({report.pass_rate:.0%})")
    print(f"Failed:             {report.failed}")
    print()
    print("AGGREGATE SCORES")
    print("-" * 40)
    print(f"Retrieval hit rate: {report.retrieval_hit_rate:.0%}  (threshold: {RETRIEVAL_HIT_THRESHOLD:.0%})")
    print(f"Avg faithfulness:   {report.avg_faithfulness:.3f}  (threshold: {FAITHFULNESS_THRESHOLD:.2f})")
    print(f"Avg relevance:      {report.avg_relevance:.3f}  (threshold: {RELEVANCE_THRESHOLD:.2f})")
    print()

    # Flag failures
    failures = [r for r in report.results if not r.passed]
    if failures:
        print("FAILED CASES")
        print("-" * 40)
        for r in failures:
            status_parts = []
            if not r.retrieval.hit:
                status_parts.append("retrieval miss")
            if r.generation.faithfulness < FAITHFULNESS_THRESHOLD:
                status_parts.append(f"faithfulness {r.generation.faithfulness:.2f}")
            if r.generation.relevance < RELEVANCE_THRESHOLD:
                status_parts.append(f"relevance {r.generation.relevance:.2f}")
            print(f"  Q: {r.question[:55]}...")
            print(f"     Failed: {', '.join(status_parts)}")
            print(f"     Reason: {r.generation.reasoning}")
            print()

    overall = "PASS" if report.retrieval_hit_rate >= RETRIEVAL_HIT_THRESHOLD else "FAIL"
    print(f"OVERALL: {overall}")
    print("=" * 65)

def main():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "rag-demo"))

    print(f"Running {len(EVAL_CASES)} evaluation cases...")
    print("Note: This uses LLM-as-judge which costs tokens. Each case makes 2 to 3 API calls.\n")

    report = run_eval(
        eval_cases=EVAL_CASES,
        index=index,
        top_k=5,
        namespace="default",
        output_file="eval_results.json",
    )

    print_report(report)
    print("\nDetailed results saved to eval_results.json")


if __name__ == "__main__":
    main()
