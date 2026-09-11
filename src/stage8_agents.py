import logging
import sys
import time
from typing import Any, Callable, TypedDict

from google import genai
from google.genai import types
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.stage1_extractor import get_document_id
from src.stage3_embed import get_db_connection, semantic_search
from src.stage4_rag import MAX_COSINE_DISTANCE, MODEL_NAME, get_gemini_client
from src.stage5_guardrails import (
    FAITHFULNESS_THRESHOLD,
    check_citations,
    evaluate_faithfulness,
)

logging.getLogger("google.genai").setLevel(logging.ERROR)

MAX_REVISION_ATTEMPTS = 1


def _call_with_retry(
    fn: Callable[[], Any],
    max_attempts: int = 4,
    base_delay: float = 3.0,
) -> Any:
    """
    Executes a model invocation with backoff.
    Detects 429 / RESOURCE_EXHAUSTED rate limits and pauses for quota reset.
    """
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            err_msg = str(e).lower()
            is_rate_limit = "429" in err_msg or "resource_exhausted" in err_msg

            if attempt == max_attempts - 1:
                raise

            # Free-tier quota reset requires longer sleep intervals
            if is_rate_limit:
                sleep_time = 25.0 + (attempt * 10.0)
                print(
                    f"[Rate Limit 429] Free-tier 5 RPM limit hit; "
                    f"cooling down for {sleep_time:.1f}s (Attempt {attempt + 1}/{max_attempts})...",
                    file=sys.stderr,
                )
            else:
                sleep_time = base_delay * (2 ** attempt)
                print(
                    f"[Retry Notice] Agent call failed ({type(e).__name__}); "
                    f"retrying in {sleep_time:.1f}s (Attempt {attempt + 1}/{max_attempts})...",
                    file=sys.stderr,
                )

            time.sleep(sleep_time)

# --- State Definition ---


class AgentGraphState(TypedDict):
    question: str
    document_id: str | None
    top_k: int
    is_financial_query: bool
    router_reasoning: str
    sources: list[dict[str, Any]]
    contexts: list[str]
    draft_answer: str
    verified_answer: str
    is_grounded: bool
    passed_guardrails: bool
    citation_audit: dict[str, Any]
    faithfulness_audit: dict[str, Any]
    critique_feedback: str | None
    retry_count: int
    max_retries: int


# --- Router Schema ---


class RouterDecision(BaseModel):
    is_financial_query: bool = Field(
        description="True if query concerns financial statements, SEC filings, regulatory disclosures, corporate governance, or business operations."
    )
    reasoning: str = Field(
        description="Concise rationale explaining the classification decision."
    )


# --- Node 1: Router Agent ---


def router_node(state: AgentGraphState, client: genai.Client) -> dict:
    """
    Classifies user intent to filter out non-financial or off-topic queries.
    Design Choice: Fails OPEN on error (defaults to True) so unexpected router
    exceptions never block valid financial questions from reaching the pipeline.
    """
    question = state["question"]

    prompt = f"""You are a query triage classifier for an SEC regulatory filings intelligence system.
Determine whether the following query is related to SEC filings, financial performance, accounting disclosures, 
corporate governance, risk factors, or corporate operations.

Query: "{question}"
"""
    try:
        response = _call_with_retry(
            lambda: client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=RouterDecision,
                ),
            )
        )
        decision = RouterDecision.model_validate_json(response.text or "{}")
        return {
            "is_financial_query": decision.is_financial_query,
            "router_reasoning": decision.reasoning,
        }
    except Exception as e:
        print(f"[Router Warning] Classification failed: {e}. Defaulting to True (fail-open).", file=sys.stderr)
        return {
            "is_financial_query": True,
            "router_reasoning": f"Router failed open due to exception: {e}",
        }


# --- Node 2: Out of Domain Node ---


def out_of_domain_node(state: AgentGraphState) -> dict:
    """Handles off-topic queries with a deterministic refusal without invoking retrieval."""
    msg = (
        "This system is specialized for SEC filings and corporate disclosures. "
        "The submitted question appears outside this domain. Please ask a financial, "
        "accounting, or regulatory question."
    )
    return {
        "draft_answer": msg,
        "verified_answer": msg,
        "is_grounded": True,
        "passed_guardrails": True,
        "citation_audit": {
            "passed": True,
            "cited_chunks": [],
            "phantom_chunks": [],
            "valid_chunks": [],
            "missing_citations": True,
            "reason": "Query routed out-of-domain.",
        },
        "faithfulness_audit": {
            "eval_status": "SUCCESS",
            "faithfulness_score": 1.0,
            "has_hallucinations": False,
            "hallucinated_claims": [],
            "reasoning": "Out-of-domain request rejected deterministically.",
            "system_error": None,
        },
    }


# --- Node 3: Retriever Agent ---


def retriever_node(state: AgentGraphState, conn) -> dict:
    """Executes vector search and isolates passing chunks below the MAX_COSINE_DISTANCE threshold."""
    query = state["question"]
    doc_id = state.get("document_id")
    top_k = state.get("top_k", 3)

    raw_matches = semantic_search(query=query, conn=conn, doc_id=doc_id, top_k=top_k)

    sources = []
    contexts = []

    for item in raw_matches:
        # item[0] is the globally unique PRIMARY KEY 'id' from Stage 3
        chunk_id, d_id, d_name, chunk_text, dist = item[0], item[1], item[2], item[3], float(item[4])
        if dist <= MAX_COSINE_DISTANCE:
            sources.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": d_id,
                    "document_name": d_name,
                    "text": chunk_text,
                    "cosine_distance": round(dist, 4),
                }
            )
            contexts.append(f"[Chunk {chunk_id}]\n{chunk_text}")

    return {
        "sources": sources,
        "contexts": contexts,
    }


# --- Node 4: Financial Analyst Agent ---


def analyst_node(state: AgentGraphState, client: genai.Client) -> dict:
    """Synthesizes an answer using context chunks and revises based on critique if retrying."""
    question = state["question"]
    contexts = state.get("contexts", [])
    critique = state.get("critique_feedback")
    retry_count = state.get("retry_count", 0)

    if not contexts:
        refusal = "The requested information is not available in the retrieved filing sections."
        return {
            "draft_answer": refusal,
            "verified_answer": refusal,
        }

    formatted_context = "\n\n---\n\n".join(contexts)

    revision_directive = ""
    if critique:
        revision_directive = f"""
IMPORTANT COMPLIANCE AUDIT FEEDBACK:
Your previous draft was rejected by the compliance auditor for the following reason:
"{critique}"
You must revise the answer to resolve this exact critique. Remove any ungrounded assertions and only cite valid chunks.
"""

    prompt = f"""You are a financial analyst specializing in SEC regulatory disclosures.
Answer the user's question using ONLY the provided filing context chunks.

Mandatory Rules:
1. Every factual statement must cite its source chunk using bracket format: [Chunk X] or [Chunks X, Y].
2. If the context does not contain enough information, state clearly what is not available.
3. Do not assume or extrapolate figures not present in the text.
{revision_directive}
Filing Context:
{formatted_context}

Question: {question}

Grounded Answer:"""

    try:
        response = _call_with_retry(
            lambda: client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                ),
            )
        )
        generated_text = (response.text or "").strip()
    except Exception as e:
        generated_text = f"Error generating analyst synthesis: {e}"

    return {
        "draft_answer": generated_text,
        "retry_count": retry_count + 1 if critique else retry_count,
    }


# --- Node 5: Compliance Auditor Agent ---


def verifier_node(state: AgentGraphState, client: genai.Client) -> dict:
    """
    Executes deterministic citation audits and faithfulness scoring.
    Design Choice: Fails CLOSED on verification errors (is_grounded: False)
    so unverified factual claims are never treated as compliant answers.
    """
    draft_answer = state["draft_answer"]
    sources = state.get("sources", [])
    contexts = state.get("contexts", [])
    question = state["question"]

    if not sources:
        return {
            "verified_answer": draft_answer,
            "is_grounded": True,
            "passed_guardrails": True,
            "citation_audit": {
                "passed": True,
                "cited_chunks": [],
                "phantom_chunks": [],
                "valid_chunks": [],
                "missing_citations": True,
                "reason": "Empty context refusal accepted.",
            },
            "faithfulness_audit": {
                "eval_status": "SUCCESS",
                "faithfulness_score": 1.0,
                "has_hallucinations": False,
                "hallucinated_claims": [],
                "reasoning": "Clean refusal on empty context.",
                "system_error": None,
            },
            "critique_feedback": None,
        }

    # Step A: Citation Syntax & Phantom Detection
    citation_audit = check_citations(draft_answer, sources)

    if not citation_audit["passed"]:
        critique = f"Citation failure: {citation_audit['reason']}"
        # verified_answer intentionally omitted here: graph topology guarantees
        # this branch routes to retry_analyst or remediate before reaching END.
        return {
            "is_grounded": False,
            "passed_guardrails": False,
            "citation_audit": citation_audit,
            "faithfulness_audit": {
                "eval_status": "SKIPPED",
                "faithfulness_score": 0.0,
                "has_hallucinations": True,
                "hallucinated_claims": ["Phantom chunk citations detected."],
                "reasoning": citation_audit["reason"],
                "system_error": None,
            },
            "critique_feedback": critique,
        }

    # Step B: LLM Faithfulness Audit
    faithfulness_audit = evaluate_faithfulness(
        question=question,
        answer=draft_answer,
        contexts=contexts,
        client=client,
    )

    eval_ok = faithfulness_audit["eval_status"] == "SUCCESS"
    score_ok = float(faithfulness_audit.get("faithfulness_score", 0.0)) >= FAITHFULNESS_THRESHOLD
    no_hallucinations = not faithfulness_audit.get("has_hallucinations", True)

    is_grounded = eval_ok and score_ok and no_hallucinations
    passed_guardrails = is_grounded

    critique = None
    if not is_grounded:
        claims = ", ".join(faithfulness_audit.get("hallucinated_claims", []))
        critique = (
            f"Faithfulness failure: Unsupported assertions detected [{claims}]. "
            f"Reasoning: {faithfulness_audit.get('reasoning')}"
        )

    return {
        "verified_answer": draft_answer,
        "is_grounded": is_grounded,
        "passed_guardrails": passed_guardrails,
        "citation_audit": citation_audit,
        "faithfulness_audit": faithfulness_audit,
        "critique_feedback": critique,
    }


# --- Node 6: Remediation Safety Node ---


def remediation_node(state: AgentGraphState) -> dict:
    """
    Fallback node triggered when self-correction retries are exhausted.
    Design Choice: Returns transparent audit metadata warning the user of
    the failure mode alongside the ungrounded draft for inspectability.
    """
    draft = state.get("draft_answer", "")
    critique = state.get("critique_feedback", "Unverified claims present.")
    sanitized = (
        f"Verification Warning: This answer failed compliance guardrails after revision.\n\n"
        f"Draft Output: {draft}\n\n"
        f"Auditor Flag: {critique}"
    )
    return {
        "verified_answer": sanitized,
    }


# --- Conditional Routing Functions ---


def route_query_decision(state: AgentGraphState) -> str:
    """Determines whether to execute retrieval or reject off-topic input."""
    if state.get("is_financial_query", True):
        return "retriever"
    return "out_of_domain"


def route_audit_decision(state: AgentGraphState) -> str:
    """Routes based on guardrail compliance and retry budgets."""
    if state.get("passed_guardrails", False):
        return "end"

    current_retries = state.get("retry_count", 0)
    max_retries = state.get("max_retries", MAX_REVISION_ATTEMPTS)

    if current_retries < max_retries:
        return "retry_analyst"
    return "remediate"


# --- Graph Construction ---


def build_agent_graph(conn, client: genai.Client):
    """Compiles the LangGraph StateGraph with dependency-injected clients."""
    workflow = StateGraph(AgentGraphState)

    # Register Nodes
    workflow.add_node("router", lambda s: router_node(s, client=client))
    workflow.add_node("out_of_domain", out_of_domain_node)
    workflow.add_node("retriever", lambda s: retriever_node(s, conn=conn))
    workflow.add_node("analyst", lambda s: analyst_node(s, client=client))
    workflow.add_node("verifier", lambda s: verifier_node(s, client=client))
    workflow.add_node("remediate", remediation_node)

    # Wire Edges
    workflow.add_edge(START, "router")

    workflow.add_conditional_edges(
        "router",
        route_query_decision,
        {
            "retriever": "retriever",
            "out_of_domain": "out_of_domain",
        },
    )

    workflow.add_edge("out_of_domain", END)
    workflow.add_edge("retriever", "analyst")
    workflow.add_edge("analyst", "verifier")

    workflow.add_conditional_edges(
        "verifier",
        route_audit_decision,
        {
            "end": END,
            "retry_analyst": "analyst",
            "remediate": "remediate",
        },
    )

    workflow.add_edge("remediate", END)

    return workflow.compile()


# --- CLI Test Runner ---

if __name__ == "__main__":
    sample_pdf_url = "https://www.sec.gov/files/form10-k.pdf"
    doc_id = get_document_id(sample_pdf_url)

    test_queries = [
        "What are the general instructions regarding Part I Item 1 business reporting?",
        "What is the best recipe for baking chocolate chip cookies?",
        "What is the CEO's personal stock option count for 2024?",
    ]

    print("--- Stage 8: Multi-Agent Orchestration Pipeline (LangGraph) ---")

    conn = get_db_connection()
    try:
        client = get_gemini_client()
        graph = build_agent_graph(conn=conn, client=client)

        for idx, q in enumerate(test_queries, start=1):
            print(f"\n[{idx}] Running Agent Workflow: '{q}'")
            initial_state: AgentGraphState = {
                "question": q,
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
                "max_retries": MAX_REVISION_ATTEMPTS,
            }

            try:
                final_state = graph.invoke(initial_state)
            except Exception as query_err:
                conn.rollback()
                print(f"[Query Error] Execution failed for test query #{idx}: {query_err}", file=sys.stderr)
                continue

            print("\nWorkflow Execution Output:")
            print(f"  * Financial Domain Match : {final_state.get('is_financial_query')}")
            print(f"  * Router Reasoning       : {final_state.get('router_reasoning')}")
            print(f"  * Passed Guardrails      : {final_state.get('passed_guardrails')}")
            print(f"  * Revision Retries Run   : {final_state.get('retry_count')}")
            print(f"  * Faithfulness Score     : {final_state.get('faithfulness_audit', {}).get('faithfulness_score')}")
            print("\nFinal Verified Response:")
            print(final_state.get("verified_answer"))
            print("-" * 70)

            # Throttle between queries to remain within 5 RPM limits during test runs
            if idx < len(test_queries):
                time.sleep(12)

        print("\nStage 8 Multi-Agent Orchestration Verified.")
    finally:
        conn.close()