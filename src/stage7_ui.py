import os
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="FinDocAI - SEC Financial Document Intelligence",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session State Initialization ---

if "last_query_result" not in st.session_state:
    st.session_state["last_query_result"] = None

if "last_query_error" not in st.session_state:
    st.session_state["last_query_error"] = None

if "ingested_docs" not in st.session_state:
    # Tracks {document_name: document_id} across successful ingests
    st.session_state["ingested_docs"] = {}


st.title("📑 FinDocAI: SEC Filing Analysis & Verification")
st.caption(
    "Interactive verification interface for grounded financial analysis with "
    "deterministic citation checks and LLM-as-a-judge faithfulness auditing."
)

# --- Sidebar: Ingestion & System Status ---

with st.sidebar:
    st.header("⚙️ System Status")

    try:
        health_resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            st.success("API Backend: Connected")
            st.markdown(f"- **Database**: `{health_data.get('database')}`")
            st.markdown(
                f"- **Gemini Client**: `{'Ready' if health_data.get('gemini_client_configured') else 'Not Ready'}`"
            )
        else:
            st.error(f"API Degraded (Status {health_resp.status_code})")
    except requests.exceptions.RequestException:
        st.error("API Backend: Offline. Ensure FastAPI is running on port 8000.")

    st.markdown("---")
    st.header("📥 Ingest SEC Filing")

    default_url = "https://www.sec.gov/files/form10-k.pdf"
    pdf_url_input = st.text_input("SEC PDF Document URL", value=default_url)

    if st.button("Download & Index Filing", use_container_width=True):
        if not pdf_url_input.strip():
            st.warning("Please provide a valid PDF URL.")
        else:
            with st.spinner("Downloading, parsing, chunking, and embedding..."):
                try:
                    ingest_payload = {"pdf_url": pdf_url_input.strip()}
                    resp = requests.post(
                        f"{API_BASE_URL}/documents/ingest",
                        json=ingest_payload,
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        doc_id = data.get("document_id")
                        doc_name = data.get("document_name")
                        st.session_state["ingested_docs"][doc_name] = doc_id

                        st.success("Document Indexed Successfully!")
                        st.markdown(f"- **Document Name**: `{doc_name}`")
                        st.markdown(f"- **Document ID**: `{doc_id}`")
                        st.markdown(f"- **Chunks Stored**: `{data.get('chunks_indexed')}`")
                    else:
                        st.error(f"Ingestion failed ({resp.status_code}): {resp.text}")
                except requests.exceptions.Timeout:
                    st.error("Ingestion timed out while extracting or generating embeddings.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error during ingestion: {e}")

    st.markdown("---")
    st.header("🔍 Retrieval Parameters")

    # Document-scoping selector populated from session ingest history
    doc_options = ["All Ingested Documents (Corpus-wide)"] + list(
        st.session_state["ingested_docs"].keys()
    )
    selected_scope = st.selectbox("Document Filter Scope", doc_options)

    if selected_scope == "All Ingested Documents (Corpus-wide)":
        target_doc_id = None
    else:
        target_doc_id = st.session_state["ingested_docs"][selected_scope]

    top_k_slider = st.slider("Context Chunks (Top K)", min_value=1, max_value=10, value=3)

# --- Main Query Panel ---

sample_queries = [
    "Select a pre-built sample query...",
    "What are the general instructions regarding Part I Item 1 business reporting?",
    "What form should be used as a blank form to be filled in according to General Instruction C?",
    "What is the CEO's personal stock option count for 2024?",
]

selected_sample = st.selectbox("Sample Test Queries", sample_queries)

query_input = st.text_area(
    "Enter Financial / Regulatory Question",
    value="" if selected_sample == sample_queries[0] else selected_sample,
    placeholder="Ask a question about filing requirements, risk factors, or governance...",
    height=90,
)

execute_btn = st.button("Run Grounded Query", type="primary", use_container_width=True)

# Process query action and write results into session state
if execute_btn:
    if not query_input.strip():
        st.warning("Please enter a question to execute.")
    else:
        with st.spinner("Retrieving contexts, generating answer, and auditing attribution..."):
            query_payload = {
                "query": query_input.strip(),
                "document_id": target_doc_id,
                "top_k": top_k_slider,
            }
            try:
                response = requests.post(
                    f"{API_BASE_URL}/query",
                    json=query_payload,
                    timeout=90,
                )
                if response.status_code == 200:
                    st.session_state["last_query_result"] = response.json()
                    st.session_state["last_query_error"] = None
                else:
                    st.session_state["last_query_result"] = None
                    st.session_state["last_query_error"] = (
                        f"Server error ({response.status_code}): {response.text}"
                    )
            except requests.exceptions.Timeout:
                st.session_state["last_query_result"] = None
                st.session_state["last_query_error"] = (
                    "Request timed out after 90 seconds. Generation or verifier retry backoff took too long."
                )
            except requests.exceptions.RequestException as req_err:
                st.session_state["last_query_result"] = None
                st.session_state["last_query_error"] = f"Backend connection failed: {req_err}"

# --- Render Persisted State ---

if st.session_state["last_query_error"]:
    st.error(st.session_state["last_query_error"])

if st.session_state["last_query_result"]:
    data = st.session_state["last_query_result"]

    verified_answer = data.get("verified_answer", "")
    is_grounded = data.get("is_grounded", False)
    passed_guardrails = data.get("passed_guardrails", False)
    citation_audit = data.get("citation_audit", {})
    faithfulness_audit = data.get("faithfulness_audit", {})
    sources = data.get("sources", [])
    eval_status = faithfulness_audit.get("eval_status", "UNKNOWN")

    st.markdown("### 📋 Synthesized Response")

    if passed_guardrails and is_grounded:
        st.success("✅ **Guardrails Passed**: Response is verified and grounded in filing context.")
    elif eval_status == "SYSTEM_ERROR":
        st.warning("⚠️ **Audit Inconclusive**: Guardrail verifier hit an external outage. Verify raw context manually.")
    else:
        st.error("🚫 **Guardrails Failed**: Hallucination or citation mismatch detected.")

    st.markdown(verified_answer)

    # --- Metrics Bar ---
    st.markdown("---")
    st.subheader("🛡️ Compliance & Attribution Audit")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Grounded", "Yes" if is_grounded else "No")
    with col2:
        if eval_status == "SYSTEM_ERROR":
            st.metric("Faithfulness Score", "N/A (Error)")
        else:
            score = faithfulness_audit.get("faithfulness_score", 0.0)
            st.metric("Faithfulness Score", f"{score * 100:.0f}%")
    with col3:
        st.metric("Cited Chunks", str(citation_audit.get("cited_chunks", [])))
    with col4:
        phantom_count = len(citation_audit.get("phantom_chunks", []))
        st.metric("Phantom Chunks", str(phantom_count))

    # --- Expanders ---
    with st.expander("🔍 Detailed Audit Trace", expanded=False):
        st.markdown(f"**Audit Status:** `{eval_status}`")
        st.markdown(f"**Faithfulness Reasoning:** {faithfulness_audit.get('reasoning')}")
        st.markdown(f"**Citation Reasoning:** {citation_audit.get('reason')}")
        if faithfulness_audit.get("hallucinated_claims"):
            st.markdown("**Flagged Claims:**")
            for claim in faithfulness_audit["hallucinated_claims"]:
                st.markdown(f"- ⚠️ {claim}")
        if faithfulness_audit.get("system_error"):
            st.markdown(f"**System Exception:** `{faithfulness_audit.get('system_error')}`")

    with st.expander("📚 Retrieved Source Chunks", expanded=False):
        if not sources:
            st.info("No source chunks retrieved.")
        else:
            for src in sources:
                st.markdown(
                    f"**Chunk ID #{src.get('chunk_id')}** — "
                    f"`{src.get('document_name')}` (Doc ID: `{src.get('document_id')}`) — "
                    f"Cosine Distance: `{src.get('cosine_distance')}`"
                )
                st.text(src.get("text", ""))
                st.divider()