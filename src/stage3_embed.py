import os
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from src.stage1_extractor import get_document_id,get_extracted_text
from src.stage2_chunking import preprocess_text,chunk_text_by_words


load_dotenv()

DATABASE_URL=os.getenv("DATABASE_URL")
EMBEDDING_MODEL_NAME="BAAI/bge-small-en-v1.5"
VECTOR_DIMENSION=384
BGE_QUERY_PREFIX="Represent this sentence for searching relevant passages: "


print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_db_connection():

    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not set in ur .env file.")
    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)
    return conn

def init_vector_table(conn):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                document_id TEXT NOT NULL,
                document_name TEXT NOT NULL,
                chunk_id INT NOT NULL,
                chunk_text TEXT NOT NULL,
                word_count INT NOT NULL,
                embedding vector({VECTOR_DIMENSION}) NOT NULL,
                CONSTRAINT unique_doc_chunk UNIQUE (document_id, chunk_id)
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx
            ON document_chunks USING hnsw (embedding vector_cosine_ops);
            """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS document_chunks_doc_id_idx
            ON document_chunks (document_id);
        """)
        conn.commit()
    print("Database Table 'document_chunks' and indexes verified.")

def generate_and_store_embeddings(
        chunks: list[dict],
        doc_id: str,
        doc_name: str,
        conn,
        model: SentenceTransformer=embedding_model,
        batch_size: int=32
):

    texts=[chunk["text"] for chunk in chunks]
    print(f"Generating embeddings for {len(texts)} chunks (batch_size={batch_size})...")
    embeddings=model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM document_chunks WHERE document_id = %s;",(doc_id,))

        insert_query="""
            INSERT INTO document_chunks (document_id, document_name,chunk_id,chunk_text,word_count,embedding)
            VALUES (%s,%s,%s,%s,%s,%s);
        """
        for chunk,emb in zip(chunks,embeddings):
            cur.execute(
                insert_query,
                (
                    doc_id,
                    doc_name,
                    chunk["chunk_id"],
                    chunk["text"],
                    chunk["word_count"],
                    emb.tolist()
                )
            )
        conn.commit()
    print(f"Successfully stored {len(chunks)} chunks for document_id '{doc_id}' in PostgreSQL!")
def semantic_search(
        query: str,
        conn,
        doc_id: str | None=None,
        top_k:int=3,
        model: SentenceTransformer=embedding_model
)-> list[tuple]:


    prefixed_query=f"{BGE_QUERY_PREFIX}{query}"
    query_vector=model.encode(prefixed_query).tolist()

    with conn.cursor() as cur:
        if doc_id:
            sql = """
                SELECT chunk_id,document_id,document_name,chunk_text,(embedding <=> %s::vector) AS distance
                FROM document_chunks
                WHERE document_id=%s
                ORDER BY distance ASC
                LIMIT %s;            
            """
            cur.execute(sql,(query_vector,doc_id,top_k))
        else:
            sql ="""
                SELECT chunk_id, document_id, document_name, chunk_text, (embedding <=> %s::vector) AS distance
                FROM document_chunks
                ORDER BY distance ASC
                LIMIT %s;  
            """
            cur.execute(sql, (query_vector,top_k))
        results=cur.fetchall()
        return results
if __name__=="__main__":
    sample_pdf_url="https://www.sec.gov/files/form10-k.pdf"
    doc_id=get_document_id(sample_pdf_url)
    doc_name="form10-k.pdf"

    print("--- Stage 3: Embedding & Vector Storage Pipeline ---")
    conn=get_db_connection()
    try:
        init_vector_table(conn)
        raw_text=get_extracted_text(sample_pdf_url)
        cleaned_text=preprocess_text(raw_text)
        chunks=chunk_text_by_words(cleaned_text,chunk_size=500,overlap=50)

        generate_and_store_embeddings(chunks, doc_id=doc_id,doc_name=doc_name,conn=conn)

        test_query="What information must be included in Part I Item 1 regarding buisness operations?"
        print(f"\n[Test Query]: '{test_query}")
        matches= semantic_search(test_query,conn=conn,doc_id=doc_id,top_k=2)
        print("\n--- Semantic Search Results (Closest Chunks) ---")
        for rank, (c_id, d_id, d_name, text, distance) in enumerate(matches, start=1):
            print(f"\n[Rank #{rank}] Chunk ID: {c_id} | Doc ID: {d_id} | Cosine Distance: {distance:.4f}")
            print(f"Content: {text[:220]}...")
            
        print("\n" + "=" * 60)
        print("Stage 3 Vector Storage & Search Complete!")
        
    finally:
        conn.close()
