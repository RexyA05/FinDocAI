import logging
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from psycopg2.extensions import connection as PgConnection

from src.stage1_extractor import get_document_id
from src.stage3_embed import get_db_connection, semantic_search

logging.getLogger("google.genai").setLevel(logging.ERROR)
load_dotenv()

MODEL_NAME = "gemini-3.6-flash"
MAX_COSINE_DISTANCE = 0.65

SYSTEM_INSTRUCTION = """You are a financial analyst assistant specializing in SEC regulatory filings.
Answer the question strictly using only the provided context. If the context does not contain enough information to answer definitively, state that the information is not available in the retrieved filing sections. Do not speculate or introduce outside knowledge.

Always cite the exact Chunk ID using bracket notation (e.g., [Chunk 12]) immediately following any fact or data point derived from that chunk."""


def get_gemini_client(api_key: str | None = None) -> genai.Client:
    """Initializes and returns the Google GenAI client with lazy key resolution."""
    raw_key = api_key or os.getenv("GEMINI_API_KEY")
    if not raw_key:
        raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")
    
    resolved_key = raw_key.strip("'\"\t\r\n")
    if not resolved_key or resolved_key.startswith("your-"):
        raise ValueError("GEMINI_API_KEY is invalid. Please check your .env file.")
        
    return genai.Client(api_key=resolved_key)


def build_rag_prompt(
    query: str,
    retrieved_chunks: list[tuple[int, str, str, str, float]],
) -> str:
    """
    Constructs the augmented context block for the model prompt.
    Keeps vector distances out of the prompt body to avoid numeric confusion.
    """
    context_blocks = []
    for chunk_id, _, doc_name, text, _ in retrieved_chunks:
        block = f"[Chunk {chunk_id}] (Document: {doc_name})\n{text}"
        context_blocks.append(block)

    formatted_context = "\n\n---\n\n".join(context_blocks)

    return f"""Retrieved Filing Context:
-------------------------
{formatted_context}
-------------------------

Question: {query}

Analytical Answer:"""


def generate_grounded_answer(
    query: str,
    conn: PgConnection,
    doc_id: str | None = None,
    top_k: int = 3,
    client: genai.Client | None = None,
    distance_threshold: float = MAX_COSINE_DISTANCE,
) -> dict:
    """
    Executes the baseline RAG pipeline:
    1. Vector retrieval via Stage 3 (guaranteed ORDER BY distance ASC).
    2. Cosine distance cutoff filtering.
    3. Guarded generation with temperature=0.0 and isolated system instruction.
    4. Returns a structured schema ready for Stage 5 guardrails, FastAPI, and RAGAS.
    """
    if client is None:
        client = get_gemini_client()

    try:
        raw_chunks = semantic_search(query=query, conn=conn, doc_id=doc_id, top_k=top_k)
    except Exception as db_err:
        conn.rollback()
        return {
            "question": query,
            "answer": f"Database retrieval error: {db_err}",
            "contexts": [],
            "sources": [],
        }

    retrieved_chunks = [
        (chunk_id, doc_id_val, doc_name, text.strip(), distance)
        for chunk_id, doc_id_val, doc_name, text, distance in raw_chunks
        if distance <= distance_threshold
    ]

    print(
        f"[Retrieval] {len(retrieved_chunks)}/{len(raw_chunks)} chunks "
        f"passed distance cutoff (<= {distance_threshold})"
    )

    if not retrieved_chunks:
        return {
            "question": query,
            "answer": "The provided filing context does not contain relevant information to answer this question.",
            "contexts": [],
            "sources": [],
        }

    sources = [
        {
            "chunk_id": chunk_id,
            "document_id": doc_id_val,
            "document_name": doc_name,
            "text": text,
            "cosine_distance": round(distance, 4),
        }
        for chunk_id, doc_id_val, doc_name, text, distance in retrieved_chunks
    ]

    prompt = build_rag_prompt(query, retrieved_chunks)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.0,
            ),
        )
        answer_text = (
            response.text
            if response.text
            else "Unable to synthesize answer from retrieved context."
        )
    except Exception as api_err:
        answer_text = f"Generation error from model provider: {api_err}"

    return {
        "question": query,
        "answer": answer_text,
        "contexts": [s["text"] for s in sources],
        "sources": sources,
    }


if __name__ == "__main__":
    sample_pdf_url = "https://www.sec.gov/files/form10-k.pdf"
    doc_id = get_document_id(sample_pdf_url)

    test_queries = [
        "What are the general instructions regarding Part I Item 1 business reporting?",
        "What form should be used as a blank form to be filled in according to General Instruction C?",
        "What is the CEO's personal stock option count for 2024?",  # Hallucination rejection test
    ]

    print("--- Stage 4: End-to-End RAG Generation Pipeline ---")

    conn = get_db_connection()
    try:
        gemini_client = get_gemini_client()

        for idx, query in enumerate(test_queries, start=1):
            print(f"\n[{idx}] User Query: {query}")
            try:
                result = generate_grounded_answer(
                    query=query,
                    conn=conn,
                    doc_id=doc_id,
                    top_k=3,
                    client=gemini_client,
                )
            except Exception as loop_err:
                conn.rollback()
                print(f"Error processing query {idx}: {loop_err}")
                continue

            print("\nGenerated Response:")
            print(result["answer"])
            print(f"\nRetrieved Chunks ({len(result['sources'])} passing threshold):")
            for src in result["sources"]:
                dist = src.get("cosine_distance", "N/A")
                print(
                    f"  * [Chunk {src['chunk_id']}] from {src['document_name']} (Distance: {dist})"
                )
            print("-" * 60)

        print("\nStage 4 RAG Pipeline Execution Complete.")
    finally:
        conn.close()