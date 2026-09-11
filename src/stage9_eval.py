import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

from google import genai
from google.genai import types

from src.stage1_extractor import get_document_id
from src.stage3_embed import get_db_connection, semantic_search
from src.stage4_rag import (
    MAX_COSINE_DISTANCE,
    MODEL_NAME,
    generate_grounded_answer,
    get_gemini_client,
)
from src.stage5_guardrails import (
    FAITHFULNESS_THRESHOLD,
    check_citations,
    evaluate_faithfulness,
)
from src.stage8_agents import AgentGraphState, build_agent_graph

logging.getLogger("google.genai").setLevel(logging.ERROR)

REPORTS_DIR = "reports"
RATE_LIMIT_DELAY = 12.0  # Seconds between model calls to respect free-tier 5 RPM limit


@dataclass
class TestCase:
    query_id: str
    query: str
    category: str  # "factual", "adversarial_negative", "out_of_domain"
    expected_chunk_ids: list[int]
    expected_refusal: bool
    reference_fact: str


GOLDEN_BENCHMARK: list[TestCase] = [
    TestCase(
        query_id="TC-01",
        query="What are the general instructions regarding Part I Item 1 business reporting?",
        category="factual",
        expected_chunk_ids=[4, 5],
        expected_refusal=False,
        reference_fact="Wholly-owned subsidiaries may furnish a brief description of business; asset-backed issuers may omit Item 1 entirely.",
    ),
    TestCase(
        query_id="TC-02",
        query="What form should be used as a blank form to be filled in according to General Instruction C?",
        category="factual",
        expected_chunk_ids=[2],
        expected_refusal=False,
        reference_fact="Form 10-K is not to be used as a blank form to be filled in, but only as a guide in copy preparation.",
    ),
    TestCase(
        query_id="TC-03",
        query="What is the CEO's personal stock option count for 2024?",
        category="adversarial_negative",
        expected_chunk_ids=[],
        expected_refusal=True,
        reference_fact="Information regarding executive personal stock options is not contained within the blank Form 10-K filing instructions.",
    ),
    TestCase(
        query_id="TC-04",
        query="What is the best recipe for baking chocolate chip cookies?",
        category="out_of_domain",
        expected_chunk_ids=[],
        expected_refusal=True,
        reference_fact="Off-topic query not related to financial disclosures or SEC filings.",
    ),
]


# --- Metric Scoring Helpers ---


def calculate_retrieval_metrics(
    retrieved_chunk_ids: list[int],
    expected_chunk_ids: list[int],
) -> tuple[float, float]:
    """Computes Context Precision and Context Recall against reference chunk targets."""
    if not expected_chunk_ids:
        # For negative/out-of-domain queries, retrieval of 0 chunks is ideal
        return 1.0, 1.0

    retrieved_set = set(retrieved_chunk_ids)
    expected_set = set(expected_chunk_ids)

    hits = retrieved_set.intersection(expected_set)
    recall = len(hits) / len(expected_set) if expected_set else 1.0
    precision = len(hits) / len(retrieved_set) if retrieved_set else 0.0

    return round(precision, 3), round(recall, 3)


def calculate_citation_precision(citation_audit: dict[str, Any]) -> float:
    """Calculates the proportion of cited chunks that are valid and non-phantom."""
    cited = citation_audit.get("cited_chunks", [])
    if not cited:
        return 1.0 if citation_audit.get("passed", False) else 0.0

    phantoms = citation_audit.get("phantom_chunks", [])
    valid_count = len(cited) - len(phantoms)
    return max(0.0, round(valid_count / len(cited), 3))


def evaluate_refusal_calibration(answer: str, expected_refusal: bool) -> bool:
    """Checks if the system correctly refused an unanswerable or out-of-scope query."""
    refusal_cues = [
        "not available",
        "not provided",
        "outside this domain",
        "not contained",
        "cannot answer",
        "no information",
    ]
    has_refusal_cues = any(cue in answer.lower() for cue in refusal_cues)

    if expected_refusal:
        return has_refusal_cues
    return not has_refusal_cues


# --- Pipeline Executors ---


def run_baseline_rag(
    test_case: TestCase,
    conn,
    client: genai.Client,
    doc_id: str,
) -> dict[str, Any]:
    """Executes Stage 4 Baseline RAG without multi-agent routing or self-correction."""
    raw_result = generate_grounded_answer(
        query=test_case.query,
        conn=conn,
        doc_id=doc_id,
        top_k=3,
        client=client,
    )

    # Deterministic citation check
    cite_audit = check_citations(raw_result["answer"], raw_result["sources"])

    # Faithfulness audit
    time.sleep(RATE_LIMIT_DELAY)
    contexts = [f"[Chunk {s['chunk_id']}]\n{s['text']}" for s in raw_result["sources"]]
    faith_audit = evaluate_faithfulness(
        question=test_case.query,
        answer=raw_result["answer"],
        contexts=contexts,
        client=client,
    )

    retrieved_ids = [s["chunk_id"] for s in raw_result["sources"]]
    prec, rec = calculate_retrieval_metrics(retrieved_ids, test_case.expected_chunk_ids)

    return {
        "pipeline": "Baseline RAG (Stage 4)",
        "answer": raw_result["answer"],
        "retrieved_chunks": retrieved_ids,
        "context_precision": prec,
        "context_recall": rec,
        "citation_precision": calculate_citation_precision(cite_audit),
        "faithfulness_score": faith_audit.get("faithfulness_score", 0.0),
        "passed_guardrails": cite_audit["passed"] and (faith_audit.get("faithfulness_score", 0.0) >= FAITHFULNESS_THRESHOLD),
        "refusal_calibrated": evaluate_refusal_calibration(raw_result["answer"], test_case.expected_refusal),
    }


def run_agentic_rag(
    test_case: TestCase,
    graph,
    doc_id: str,
) -> dict[str, Any]:
    """Executes Stage 8 LangGraph Multi-Agent Orchestration with autonomous critique loops."""
    initial_state: AgentGraphState = {
        "question": test_case.query,
        "document_id": doc_id,
        "top_k": 3,
        "is_financial_query": True,
        "router_reasoning": "",
        "sources": [],
        "contexts": [],
        "draft_answer": "",
        "verified_answer": "",
        "is_grounded": False,
        "passed_guardrails": False,
        "citation_audit": {},
        "faithfulness_audit": {},
        "critique_feedback": None,
        "retry_count": 0,
        "max_retries": 1,
    }

    final_state = graph.invoke(initial_state)

    retrieved_ids = [s["chunk_id"] for s in final_state.get("sources", [])]
    prec, rec = calculate_retrieval_metrics(retrieved_ids, test_case.expected_chunk_ids)
    cite_audit = final_state.get("citation_audit", {})
    faith_audit = final_state.get("faithfulness_audit", {})
    answer = final_state.get("verified_answer", "")

    return {
        "pipeline": "Agentic Graph (Stage 8)",
        "answer": answer,
        "retrieved_chunks": retrieved_ids,
        "context_precision": prec,
        "context_recall": rec,
        "citation_precision": calculate_citation_precision(cite_audit),
        "faithfulness_score": faith_audit.get("faithfulness_score", 0.0),
        "passed_guardrails": final_state.get("passed_guardrails", False),
        "refusal_calibrated": evaluate_refusal_calibration(answer, test_case.expected_refusal),
        "retries_invoked": final_state.get("retry_count", 0),
        "routed_out_of_domain": not final_state.get("is_financial_query", True),
    }


# --- Evaluation Harness ---


def run_benchmark_suite(sample_pdf_url: str = "https://www.sec.gov/files/form10-k.pdf"):
    """Runs comparative evaluation across all test cases and exports quantitative reports."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    doc_id = get_document_id(sample_pdf_url)

    conn = get_db_connection()
    client = get_gemini_client()

    print("=" * 80)
    print("FIN-DOC-AI: AUTOMATED BENCHMARKING & EVALUATION HARNESS (STAGE 9)")
    print("=" * 80)
    print(f"Target Document ID: {doc_id}")
    print(f"Total Test Cases  : {len(GOLDEN_BENCHMARK)}")
    print("Comparing         : Baseline RAG (Stage 4) vs. Multi-Agent Graph (Stage 8)")
    print("-" * 80)

    try:
        graph = build_agent_graph(conn=conn, client=client)
        results = []

        for idx, tc in enumerate(GOLDEN_BENCHMARK, start=1):
            print(f"\n[Case {idx}/{len(GOLDEN_BENCHMARK)}] Query ID: {tc.query_id} ({tc.category.upper()})")
            print(f"Query: \"{tc.query}\"")

            # 1. Run Baseline RAG
            print("  -> Executing Baseline RAG (Stage 4)...")
            try:
                base_metrics = run_baseline_rag(tc, conn=conn, client=client, doc_id=doc_id)
            except Exception as e:
                conn.rollback()
                print(f"  [!] Baseline failed: {e}", file=sys.stderr)
                base_metrics = {"error": str(e), "pipeline": "Baseline RAG (Stage 4)"}

            # Pace between pipeline executions to prevent free-tier 429 errors
            time.sleep(RATE_LIMIT_DELAY)

            # 2. Run Agentic RAG
            print("  -> Executing Multi-Agent Graph (Stage 8)...")
            try:
                agent_metrics = run_agentic_rag(tc, graph=graph, doc_id=doc_id)
            except Exception as e:
                conn.rollback()
                print(f"  [!] Agentic failed: {e}", file=sys.stderr)
                agent_metrics = {"error": str(e), "pipeline": "Agentic Graph (Stage 8)"}

            case_entry = {
                "test_case": asdict(tc),
                "baseline_evaluation": base_metrics,
                "agentic_evaluation": agent_metrics,
            }
            results.append(case_entry)

            # Intermediate print
            print(f"  Baseline Faithfulness: {base_metrics.get('faithfulness_score', 'N/A')} | Refusal OK: {base_metrics.get('refusal_calibrated')}")
            print(f"  Agentic  Faithfulness: {agent_metrics.get('faithfulness_score', 'N/A')} | Refusal OK: {agent_metrics.get('refusal_calibrated')}")

            if idx < len(GOLDEN_BENCHMARK):
                time.sleep(RATE_LIMIT_DELAY)

        # --- Aggregate Metric Computation ---
        def avg(lst):
            return round(sum(lst) / len(lst), 3) if lst else 0.0

        valid_base = [r["baseline_evaluation"] for r in results if "error" not in r["baseline_evaluation"]]
        valid_agent = [r["agentic_evaluation"] for r in results if "error" not in r["agentic_evaluation"]]

        summary = {
            "total_queries": len(GOLDEN_BENCHMARK),
            "baseline": {
                "avg_faithfulness": avg([m["faithfulness_score"] for m in valid_base]),
                "avg_context_precision": avg([m["context_precision"] for m in valid_base]),
                "avg_context_recall": avg([m["context_recall"] for m in valid_base]),
                "avg_citation_precision": avg([m["citation_precision"] for m in valid_base]),
                "guardrail_pass_rate": avg([1.0 if m["passed_guardrails"] else 0.0 for m in valid_base]),
                "refusal_accuracy": avg([1.0 if m["refusal_calibrated"] else 0.0 for m in valid_base]),
            },
            "agentic": {
                "avg_faithfulness": avg([m["faithfulness_score"] for m in valid_agent]),
                "avg_context_precision": avg([m["context_precision"] for m in valid_agent]),
                "avg_context_recall": avg([m["context_recall"] for m in valid_agent]),
                "avg_citation_precision": avg([m["citation_precision"] for m in valid_agent]),
                "guardrail_pass_rate": avg([1.0 if m["passed_guardrails"] else 0.0 for m in valid_agent]),
                "refusal_accuracy": avg([1.0 if m["refusal_calibrated"] else 0.0 for m in valid_agent]),
            },
        }

        # --- Export JSON Report ---
        report_data = {
            "summary_metrics": summary,
            "detailed_results": results,
        }
        report_path = os.path.join(REPORTS_DIR, "eval_benchmark_results.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # --- Render Terminal Benchmark Summary ---
        print("\n" + "=" * 80)
        print("FINAL BENCHMARK COMPARATIVE SUMMARY")
        print("=" * 80)
        print(f"{'Metric':<28} | {'Baseline (Stage 4)':<20} | {'Agentic (Stage 8)':<20}")
        print("-" * 80)
        print(f"{'Avg Faithfulness Score':<28} | {summary['baseline']['avg_faithfulness']:<20} | {summary['agentic']['avg_faithfulness']:<20}")
        print(f"{'Context Precision':<28} | {summary['baseline']['avg_context_precision']:<20} | {summary['agentic']['avg_context_precision']:<20}")
        print(f"{'Context Recall':<28} | {summary['baseline']['avg_context_recall']:<20} | {summary['agentic']['avg_context_recall']:<20}")
        print(f"{'Citation Precision':<28} | {summary['baseline']['avg_citation_precision']:<20} | {summary['agentic']['avg_citation_precision']:<20}")
        print(f"{'Guardrail Pass Rate':<28} | {summary['baseline']['guardrail_pass_rate'] * 100:<19.1f}% | {summary['agentic']['guardrail_pass_rate'] * 100:<19.1f}%")
        print(f"{'Negative Refusal Accuracy':<28} | {summary['baseline']['refusal_accuracy'] * 100:<19.1f}% | {summary['agentic']['refusal_accuracy'] * 100:<19.1f}%")
        print("=" * 80)
        print(f"Detailed JSON benchmark report written to: {report_path}\n")

    finally:
        conn.close()


if __name__ == "__main__":
    run_benchmark_suite()