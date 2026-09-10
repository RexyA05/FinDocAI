import logging
import re
import sys
import time
from typing import Callable
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from src.stage3_embed import get_db_connection
from src.stage4_rag import (
    MODEL_NAME,
    generate_grounded_answer,
    get_gemini_client,
)
from src.stage1_extractor import get_document_id


logging.getLogger("google.genai").setLevel(logging.ERROR)

FAITHFULNESS_THRESHOLD = 0.80

# Matches both singular and plural bracket citations: [Chunk 1], [Chunks 2, 3], [chunk 4, Chunk 5]
CITATION_REGEX = re.compile(
    r"\[(?:Chunks?|Chunk)\s*\d+(?:\s*,\s*(?:Chunks?|Chunk)?\s*\d+)*\]",
    re.IGNORECASE,
)

VERIFIER_SYSTEM_INSTRUCTION = """You are an SEC compliance auditor and fact-checker.
Your job is to strictly evaluate whether a generated answer is fully grounded in the provided filing context chunks.

Audit Rules:
1. Examine every factual assertion in the answer.
2. Check if the assertion is directly and explicitly supported by the context.
3. Identify any unsupported claims, numerical discrepancies, or external knowledge leakage.
4. If the answer states that information is not available, verify whether the context indeed lacks that information.
5. Return your evaluation strictly formatted according to the required schema."""


class FaithfulnessReport(BaseModel):
    faithfulness_score: float = Field(
        description="Float between 0.0 and 1.0 representing the proportion of assertions directly supported by the context."
    )
    has_hallucinations: bool = Field(
        description="True if any assertion in the answer is not supported by the retrieved context."
    )
    hallucinated_claims: list[str] = Field(
        default_factory=list,
        description="List of specific sentences, figures, or claims not supported by the context.",
    )
    reasoning: str = Field(
        description="Detailed analytical explanation of the verification audit."
    )


def _call_with_retry(
    fn: Callable, max_attempts: int = 3, base_delay: float = 2.0
):
    """Executes a callable with exponential backoff for transient provider spikes (e.g., 503)."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            sleep_time = base_delay * (2 ** attempt)
            print(
                f"[Retry Notice] Provider call failed ({type(e).__name__}); "
                f"retrying in {sleep_time:.1f}s (Attempt {attempt + 1}/{max_attempts})...",
                file=sys.stderr,
            )
            time.sleep(sleep_time)


def extract_cited_chunks(answer_text: str) -> list[int]:
    """
    Extracts all chunk IDs cited in brackets, handling singular, plural,
    and comma-delimited forms:
    '[Chunk 5]' -> [5]
    '[Chunks 0, Chunk 1]' -> [0, 1]
    '[chunks 2, 3]' -> [2, 3]
    """
    bracket_matches = CITATION_REGEX.findall(answer_text)
    found_ids = set()
    for match in bracket_matches:
        digits = re.findall(r"\b\d+\b", match)
        for digit in digits:
            found_ids.add(int(digit))
    return sorted(list(found_ids))


def check_citations(answer: str, sources: list[dict]) -> dict:
    """
    Validates emitted citations against actual retrieved chunk IDs.
    Detects phantom citations (citing chunks that were never retrieved).
    
    Missing citations are NOT treated as an immediate failure here:
    a valid refusal legitimately cites nothing. That case is deferred
    to the LLM faithfulness judge to evaluate claim grounding.
    """
    retrieved_chunk_ids = {s["chunk_id"] for s in sources}
    cited_chunk_ids = set(extract_cited_chunks(answer))

    phantom_citations = cited_chunk_ids - retrieved_chunk_ids
    valid_citations = cited_chunk_ids & retrieved_chunk_ids
    missing_citations = not cited_chunk_ids

    passed = not phantom_citations

    if phantom_citations:
        reason = (
            f"Phantom citations detected: cited Chunk IDs "
            f"{sorted(list(phantom_citations))} were not retrieved."
        )
    elif missing_citations:
        reason = (
            "No bracketed chunk citations found; deferring to the "
            "faithfulness audit to verify whether any unsupported claims were made."
        )
    else:
        reason = "All citations are valid and present in retrieved context."

    return {
        "passed": passed,
        "cited_chunks": sorted(list(cited_chunk_ids)),
        "phantom_chunks": sorted(list(phantom_citations)),
        "valid_chunks": sorted(list(valid_citations)),
        "missing_citations": missing_citations,
        "reason": reason,
    }


def evaluate_faithfulness(
    question: str,
    answer: str,
    contexts: list[str],
    client: genai.Client,
) -> dict:
    """
    Calls Gemini using Pydantic schema validation and retry backoff to verify grounding.
    Separates runtime/system execution failures from actual verified hallucinations.
    """
    joined_context = "\n\n---\n\n".join(contexts)

    audit_prompt = f"""Evaluate whether the following generated answer is strictly grounded in the retrieved filing context.

Retrieved Filing Context:
-------------------------
{joined_context}
-------------------------

Question: {question}

Generated Answer: {answer}"""

    try:
        response = _call_with_retry(
            lambda: client.models.generate_content(
                model=MODEL_NAME,
                contents=audit_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=VERIFIER_SYSTEM_INSTRUCTION,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=FaithfulnessReport,
                ),
            )
        )

        raw_text = response.text or ""
        parsed = FaithfulnessReport.model_validate_json(raw_text)

        return {
            "eval_status": "SUCCESS",
            "faithfulness_score": float(parsed.faithfulness_score),
            "has_hallucinations": bool(parsed.has_hallucinations),
            "hallucinated_claims": parsed.hallucinated_claims,
            "reasoning": parsed.reasoning,
            "system_error": None,
        }

    except Exception as eval_err:
        err_type = type(eval_err).__name__
        print(f"[Warning] Faithfulness evaluation failed ({err_type}): {eval_err}", file=sys.stderr)
        return {
            "eval_status": "SYSTEM_ERROR",
            "faithfulness_score": 0.0,
            "has_hallucinations": False,
            "hallucinated_claims": [],
            "reasoning": f"Automated audit could not execute: [{err_type}] {eval_err}",
            "system_error": f"[{err_type}] {eval_err}",
        }


def apply_guardrails(
    rag_result: dict,
    client: genai.Client | None = None,
    faithfulness_threshold: float = FAITHFULNESS_THRESHOLD,
) -> dict:
    """
    Executes the full Stage 5 verification workflow on a Stage 4 RAG result:
    1. Deterministic citation validation (rejects phantom citations).
    2. Faithfulness evaluation via LLM-as-a-judge with retry protection.
    3. Remediation fallback if verification fails or system errors occur.
    """
    if client is None:
        client = get_gemini_client()

    question = rag_result.get("question", "")
    raw_answer = rag_result.get("answer", "")
    sources = rag_result.get("sources", [])
    contexts = rag_result.get("contexts", [])

    # Case A: Pure empty context retrieval
    if not sources:
        return {
            "question": question,
            "raw_answer": raw_answer,
            "verified_answer": raw_answer,
            "is_grounded": True,
            "passed_guardrails": True,
            "citation_audit": {
                "passed": True,
                "cited_chunks": [],
                "phantom_chunks": [],
                "valid_chunks": [],
                "missing_citations": True,
                "reason": "No context was retrieved; answer is an ungrounded direct refusal.",
            },
            "faithfulness_audit": {
                "eval_status": "SUCCESS",
                "faithfulness_score": 1.0,
                "has_hallucinations": False,
                "hallucinated_claims": [],
                "reasoning": "Zero sources retrieved; empty context refusal accepted without LLM audit.",
                "system_error": None,
            },
            "sources": [],
        }

    # Step 1: Citation Syntax & Existence Check
    citation_audit = check_citations(raw_answer, sources)

    # Reject immediately ONLY if phantom citations are present
    if not citation_audit["passed"]:
        sanitized_answer = (
            "Verification Failed: The generated response cited sources that were not "
            "present in the retrieved filing context."
        )
        return {
            "question": question,
            "raw_answer": raw_answer,
            "verified_answer": sanitized_answer,
            "is_grounded": False,
            "passed_guardrails": False,
            "citation_audit": citation_audit,
            "faithfulness_audit": {
                "eval_status": "SKIPPED",
                "faithfulness_score": 0.0,
                "has_hallucinations": True,
                "hallucinated_claims": ["Phantom chunk references detected."],
                "reasoning": citation_audit["reason"],
                "system_error": None,
            },
            "sources": sources,
        }

    # Step 2: Faithfulness Evaluation via LLM Judge
    faithfulness_audit = evaluate_faithfulness(
        question=question,
        answer=raw_answer,
        contexts=contexts,
        client=client,
    )

    eval_ok = faithfulness_audit["eval_status"] == "SUCCESS"
    score_ok = float(faithfulness_audit.get("faithfulness_score", 0.0)) >= faithfulness_threshold
    no_hallucinations = not faithfulness_audit.get("has_hallucinations", True)

    is_grounded = eval_ok and score_ok and no_hallucinations
    passed_guardrails = is_grounded

    if is_grounded:
        verified_answer = raw_answer
    elif faithfulness_audit["eval_status"] == "SYSTEM_ERROR":
        verified_answer = (
            "Verification Inconclusive: System guardrail evaluation encountered an error. "
            "Inspect raw response with caution."
        )
    else:
        verified_answer = (
            "Verification Warning: One or more assertions in this answer could not be verified "
            "against the cited filing chunks. Please inspect retrieved sources directly."
        )

    return {
        "question": question,
        "raw_answer": raw_answer,
        "verified_answer": verified_answer,
        "is_grounded": is_grounded,
        "passed_guardrails": passed_guardrails,
        "citation_audit": citation_audit,
        "faithfulness_audit": faithfulness_audit,
        "sources": sources,
    }


if __name__ == "__main__":
    sample_pdf_url = "https://www.sec.gov/files/form10-k.pdf"
    doc_id = get_document_id(sample_pdf_url)

    test_queries = [
        "What are the general instructions regarding Part I Item 1 business reporting?",
        "What form should be used as a blank form to be filled in according to General Instruction C?",
        "What is the CEO's personal stock option count for 2024?",  # Negative refusal test
    ]

    print("--- Stage 5: Guardrails & Attribution Verification Pipeline ---")

    conn = get_db_connection()
    try:
        gemini_client = get_gemini_client()

        for idx, query in enumerate(test_queries, start=1):
            print(f"\n[{idx}] Testing Query: {query}")

            try:
                # 1. Generate via Stage 4
                rag_output = generate_grounded_answer(
                    query=query,
                    conn=conn,
                    doc_id=doc_id,
                    top_k=3,
                    client=gemini_client,
                )

                # 2. Verify via Stage 5
                verified_output = apply_guardrails(rag_output, client=gemini_client)

            except Exception as query_err:
                conn.rollback()
                print(f"Error evaluating query {idx}: {query_err}", file=sys.stderr)
                continue

            print("\nVerified Answer:")
            print(verified_output["verified_answer"])

            print("\nGuardrail Metrics:")
            print(f"  * Passed Guardrails  : {verified_output['passed_guardrails']}")
            print(f"  * Is Grounded        : {verified_output['is_grounded']}")
            print(f"  * Evaluation Status  : {verified_output['faithfulness_audit']['eval_status']}")
            print(f"  * Cited Chunks       : {verified_output['citation_audit']['cited_chunks']}")
            print(f"  * Phantom Chunks     : {verified_output['citation_audit']['phantom_chunks']}")
            print(f"  * Faithfulness Score : {verified_output['faithfulness_audit']['faithfulness_score']}")
            print(f"  * Audit Reasoning    : {verified_output['faithfulness_audit']['reasoning']}")
            if verified_output["faithfulness_audit"].get("system_error"):
                print(f"  * System Error       : {verified_output['faithfulness_audit']['system_error']}")
            print("-" * 65)

        print("\nStage 5 Guardrails Verification Pipeline Complete.")
    finally:
        conn.close()