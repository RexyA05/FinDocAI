import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Annotated, Generator

from fastapi import Depends, FastAPI, HTTPException, status
from google import genai
from psycopg2.extensions import connection as PgConnection
from pydantic import BaseModel, Field
import uvicorn

from src.stage1_extractor import (
    download_sec_pdf,
    extract_text_from_pdf_bytes,
    get_document_id,
)
from src.stage2_chunking import chunk_text_by_words, preprocess_text
from src.stage3_embed import (
    generate_and_store_embeddings,
    get_db_connection,
    init_vector_table,
)
from src.stage4_rag import generate_grounded_answer, get_gemini_client
from src.stage5_guardrails import apply_guardrails

logging.getLogger("google.genai").setLevel(logging.ERROR)

# --- Global State & Lifespan ---

app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes runtime resources on startup:
    1. Instantiates the Gemini client.
    2. Runs idempotent DB schema migrations (CREATE EXTENSION, tables, indexes).
    """
    try:
        app_state["gemini_client"] = get_gemini_client()
    except Exception as e:
        print(f"[Startup Warning] Gemini client initialization failed: {e}", file=sys.stderr)
        app_state["gemini_client"] = None

    try:
        conn = get_db_connection()
        try:
            init_vector_table(conn)
        finally:
            conn.close()
    except Exception as e:
        print(f"[Startup Error] Database initialization failed: {e}", file=sys.stderr)

    yield
    app_state.clear()


app = FastAPI(
    title="FinDocAI API",
    description="Production-ready RAG API with automated guardrails for SEC regulatory filings.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Dependency Injection ---


def get_db() -> Generator[PgConnection, None, None]:
    """
    Provides a transactional database connection per request with isolated error handling.
    Allows deliberate endpoint HTTPExceptions to bubble up unmasked.
    """
    try:
        conn = get_db_connection()
    except Exception as conn_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connection error: {conn_err}",
        )

    try:
        yield conn
    except HTTPException:
        # Deliberately raised endpoint exceptions bubble through untouched
        raise
    except Exception:
        # Roll back uncommitted transactional operations on unexpected errors
        conn.rollback()
        raise
    finally:
        conn.close()


def get_llm_client() -> genai.Client:
    """Retrieves the cached Gemini client or instantiates one."""
    client = app_state.get("gemini_client")
    if client is None:
        try:
            client = get_gemini_client()
            app_state["gemini_client"] = client
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to configure Gemini client: {e}",
            )
    return client


# --- Request & Response Schemas ---


class HealthResponse(BaseModel):
    status: str
    database: str
    gemini_client_configured: bool


class IngestRequest(BaseModel):
    pdf_url: str = Field(..., description="Publicly accessible URL to an SEC filing PDF.")


class IngestResponse(BaseModel):
    status: str
    document_id: str
    document_name: str
    chunks_indexed: int


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Financial or regulatory question.")
    document_id: str | None = Field(
        default=None,
        description="Optional doc_id filter to scope retrieval to a single document.",
    )
    top_k: int = Field(default=3, ge=1, le=10, description="Number of context chunks to retrieve.")


class SourceChunk(BaseModel):
    chunk_id: int
    document_id: str
    document_name: str
    text: str
    cosine_distance: float


class CitationAudit(BaseModel):
    passed: bool
    cited_chunks: list[int]
    phantom_chunks: list[int]
    valid_chunks: list[int]
    missing_citations: bool
    reason: str


class FaithfulnessAudit(BaseModel):
    eval_status: str
    faithfulness_score: float
    has_hallucinations: bool
    hallucinated_claims: list[str]
    reasoning: str
    system_error: str | None = None


class QueryResponse(BaseModel):
    question: str
    verified_answer: str
    raw_answer: str
    is_grounded: bool
    passed_guardrails: bool
    citation_audit: CitationAudit
    faithfulness_audit: FaithfulnessAudit
    sources: list[SourceChunk]


# --- Endpoints ---
@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Welcome to the FinDocAI API",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check(conn: Annotated[PgConnection, Depends(get_db)]):
    ...

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check(conn: Annotated[PgConnection, Depends(get_db)]):
    """Health check endpoint verifying database connectivity and Gemini readiness."""
    db_status = "healthy"
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
    except Exception as db_err:
        print(f"[Health Check DB Error] {db_err}", file=sys.stderr)
        db_status = "unhealthy"

    return HealthResponse(
        status="ok" if db_status == "healthy" else "degraded",
        database=db_status,
        gemini_client_configured=app_state.get("gemini_client") is not None,
    )


@app.post("/documents/ingest", response_model=IngestResponse, tags=["Pipeline"])
def ingest_document(
    request: IngestRequest,
    conn: Annotated[PgConnection, Depends(get_db)],
):
    """
    Downloads an SEC filing PDF, extracts text, cleans it, chunks it,
    computes embeddings, and indexes them into pgvector via Stage 3.
    """
    doc_id = get_document_id(request.pdf_url)
    doc_name = os.path.basename(request.pdf_url.split("?")[0]) or f"doc_{doc_id}.pdf"

    # Step 1: Download & Extract
    try:
        pdf_bytes = download_sec_pdf(request.pdf_url)
        extracted_text = extract_text_from_pdf_bytes(pdf_bytes)
    except Exception as extract_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Extraction failed for URL {request.pdf_url}: {extract_err}",
        )

    if not extracted_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The supplied document yielded no extractable text.",
        )

    # Step 2: Preprocess and Chunk (produces dicts with chunk_id, text, word_count)
    cleaned_text = preprocess_text(extracted_text)
    chunks = chunk_text_by_words(cleaned_text, chunk_size=500, overlap=50)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Failed to generate chunks from the extracted text.",
        )

    # Step 3: Embed & Store in pgvector via Stage 3
    try:
        generate_and_store_embeddings(
            chunks=chunks,
            doc_id=doc_id,
            doc_name=doc_name,
            conn=conn,
        )
    except Exception as store_err:
        conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database indexing failed: {store_err}",
        )

    return IngestResponse(
        status="success",
        document_id=doc_id,
        document_name=doc_name,
        chunks_indexed=len(chunks),
    )


@app.post("/query", response_model=QueryResponse, tags=["Pipeline"])
def query_rag(
    request: QueryRequest,
    conn: Annotated[PgConnection, Depends(get_db)],
    client: Annotated[genai.Client, Depends(get_llm_client)],
):
    """
    Executes grounded RAG retrieval, Gemini generation, and Stage 5 Guardrail verification.
    """
    # Stage 4: Grounded Retrieval & Answer Generation
    try:
        rag_result = generate_grounded_answer(
            query=request.query,
            conn=conn,
            doc_id=request.document_id,
            top_k=request.top_k,
            client=client,
        )
    except Exception as gen_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG generation failure: {gen_err}",
        )

    # Stage 5: Guardrails & Attribution Audit
    try:
        verified_result = apply_guardrails(rag_result, client=client)
    except Exception as audit_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Guardrails verification failure: {audit_err}",
        )

    return QueryResponse(
        question=verified_result["question"],
        verified_answer=verified_result["verified_answer"],
        raw_answer=verified_result["raw_answer"],
        is_grounded=verified_result["is_grounded"],
        passed_guardrails=verified_result["passed_guardrails"],
        citation_audit=CitationAudit(**verified_result["citation_audit"]),
        faithfulness_audit=FaithfulnessAudit(**verified_result["faithfulness_audit"]),
        sources=[SourceChunk(**src) for src in verified_result["sources"]],
    )


if __name__ == "__main__":
    uvicorn.run("src.stage6_api:app", host="127.0.0.1", port=8000, reload=True)