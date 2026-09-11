<div align="center">

# 📑 FinDocAI

### SEC Filing Intelligence & Verification Pipeline

*A grounded RAG + multi-agent system that won't tell you something the filing doesn't say.*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API_Layer-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?style=flat-square&logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-1C3C3C?style=flat-square)](https://www.langchain.com/langgraph)
[![Gemini](https://img.shields.io/badge/Gemini-Grounded_Generation-8E75B2?style=flat-square&logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Active_Development-yellow?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)]()

</div>

FinDocAI is a Retrieval-Augmented Generation (RAG) and multi-agent system for automated, source-grounded analysis of SEC Form 10-K filings. It is built around a two-tier auditing layer — **deterministic citation verification** combined with **LLM-as-a-judge faithfulness scoring** — orchestrated via an autonomous **LangGraph self-correction loop**, with the goal of catching unsupported claims before they reach the user.

> 🎓 **Portfolio project note:** this is a solo-built learning project (3rd-year CS/DS), not a production system. The [Known Limitations](#-known-limitations) section below is intentionally candid — it's part of the point.

---

## 📖 Table of Contents

- [Key Highlights](#-key-highlights)
- [Architectural Overview](#-architectural-overview)
- [Pipeline Stages](#-pipeline-stages)
- [Quantitative Evaluation & Benchmarks](#-quantitative-evaluation--benchmarks)
- [Known Limitations](#-known-limitations)
- [Getting Started](#-getting-started)
- [Usage Guide](#-usage-guide)
- [Technical Design Decisions](#-technical-design-decisions)
- [Roadmap](#-roadmap)

---

## ✨ Key Highlights

| | |
|---|---|
| 🔍 | **Deterministic Citation Audit** — Extracts every bracketed citation (e.g., `[Chunk 5]`) from generated answers and cross-references it against the chunk IDs actually retrieved from PostgreSQL, flagging phantom citations that were never retrieved. |
| ⚖️ | **LLM-as-a-Judge Faithfulness Scoring** — Evaluates each generated claim against the retrieved source chunks via structured JSON output (Pydantic-schema-enforced), separating genuine faithfulness failures from system/provider errors. |
| 🔁 | **Self-Correcting Multi-Agent Graph** — LangGraph orchestrates a Query Router, Retriever, Financial Analyst, and Compliance Auditor, with a feedback loop that feeds the auditor's specific rejection reason back into the analyst's revision prompt. |
| ⚡ | **Asymmetric Failure Handling** — The query router fails *open* on classification errors (avoids rejecting a possibly-valid question due to a transient error); the faithfulness auditor fails *closed* on evaluation errors (avoids passing an answer that couldn't actually be verified). |
| 🎯 | **Document-Scoped Retrieval** — Vector search can be scoped to a single ingested filing via its document ID. *(Corpus-wide search across multiple documents is implemented but has a known chunk-identity limitation — see [Known Limitations](#-known-limitations).)* |
| 🌐 | **FastAPI Service Layer** — REST API with dependency-injected DB/LLM clients, per-request connection lifecycle management, and auto-generated OpenAPI docs. |

---

## 🏗️ Architectural Overview

```
                            ┌───────────────────────────────────┐
                            │    User Query / REST API Call     │
                            └─────────────────┬─────────────────┘
                                              │
                                              ▼
                            ┌───────────────────────────────────┐
                            │        Router Agent (Triage)      │
                            └─────────┬───────────────┬─────────┘
                 (Off-Topic Query)    │               │  (Financial / Regulatory)
               ┌──────────────────────┘               ▼
               │                    ┌───────────────────────────────────┐
               │                    │     Dense Vector Retrieval        │
               │                    │   (pgvector + HNSW Indexing)      │
               │                    └─────────────────┬─────────────────┘
               │                                      │
               │                                      ▼
               │                    ┌───────────────────────────────────┐◄──────┐
               │                    │     Financial Analyst Agent       │       │
               │                    │  (Context Synthesis & Citations)  │       │ (Self-Correction
               │                    └─────────────────┬─────────────────┘       │  Critique Loop)
               │                                      │                         │
               │                                      ▼                         │
               │                    ┌───────────────────────────────────┐       │
               │                    │     Compliance Auditor Agent      │───────┘
               │                    │ (Citation Syntax + Judge Scoring) │ (Rejected & Retries Left)
               │                    └─────────┬───────────────┬─────────┘
               │         (Audited & Approved) │               │ (Retries Exhausted)
               │       ┌──────────────────────┘               ▼
               │       │                    ┌───────────────────────────────────┐
               │       │                    │      Remediation Safety Node      │
               │       │                    └─────────────────┬─────────────────┘
               ▼       ▼                                      │
         ┌──────────────────────────────────┐                 │
         │   FastAPI / Streamlit Interface  │◄────────────────┘
         └──────────────────────────────────┘
```

---

## 🧩 Pipeline Stages

FinDocAI is organized into 9 distinct, testable modules:

| # | Stage | Module | Key Responsibilities |
|:---:| --- | --- | --- |
| 📥 | **Extraction** | `src/stage1_extractor.py` | SEC EDGAR PDF download, `pypdf`-based text extraction, URL-hash-keyed caching with a `force_download` override. |
| ✂️ | **Chunking** | `src/stage2_chunking.py` | Text cleaning and word-based sliding-window chunking (default `chunk_size=500`, `overlap=50`). |
| 🧠 | **Embedding** | `src/stage3_embed.py` | Dense vector generation via `BAAI/bge-small-en-v1.5` (384-dim), PostgreSQL `pgvector` persistence, HNSW indexing. |
| 💬 | **RAG** | `src/stage4_rag.py` | Grounded answer generation using Gemini, strict citation syntax enforcement, cosine-distance cutoff (≤ 0.65). |
| 🛡️ | **Guardrails** | `src/stage5_guardrails.py` | Deterministic phantom-citation detection, LLM-as-a-judge faithfulness scoring, refusal-aware guardrail logic. |
| 🌐 | **API** | `src/stage6_api.py` | FastAPI service with dependency injection, per-request DB connection lifecycle, and OpenAPI schemas. |
| 🖥️ | **UI** | `src/stage7_ui.py` | Streamlit analytical interface for ingestion, querying, and audit-trail visualization. |
| 🤖 | **Agents** | `src/stage8_agents.py` | LangGraph multi-agent state machine with a self-correction revision loop and provider retry/backoff logic. |
| 📊 | **Evaluation** | `src/stage9_eval.py` | Custom quantitative benchmarking harness comparing Baseline RAG (Stage 4) against Agentic Guardrailed RAG (Stage 8). |

> **Note on evaluation methodology:** Stage 9 is a hand-built harness inspired by RAGAS-style metrics (context precision/recall, citation precision, faithfulness), not the `ragas` library itself. Metric definitions and thresholds are project-specific and documented in `stage9_eval.py`.

---

## 📊 Quantitative Evaluation & Benchmarks

The pipeline was benchmarked on a small, hand-curated golden set (**n = 4 test cases**) spanning three categories: **Factual Reporting**, **Adversarial Negative Refusal**, and **Out-of-Domain Triage**.

```
================================================================================
FINAL BENCHMARK COMPARATIVE SUMMARY (STAGE 9)
================================================================================
Metric                       | Baseline (Stage 4)   | Agentic (Stage 8)    
--------------------------------------------------------------------------------
Avg Faithfulness Score       | 1.0                  | 1.0                  
Context Precision            | 0.75                 | 0.75                 
Context Recall               | 1.0                  | 1.0                  
Citation Precision           | 1.0                  | 1.0                  
Guardrail Pass Rate          | 100.0 %              | 100.0 %              
Negative Refusal Accuracy    | 100.0 %              | 100.0 %              
================================================================================
```

### 🔎 Reading these numbers honestly

> ⚠️ **Sample size:** n = 4 is a demonstration set, not a statistically powered evaluation. Treat these as illustrative, not conclusive.

> ⚠️ **Context Precision/Recall are inflated by construction:** the current scoring function assigns a perfect (1.0, 1.0) to both negative and out-of-domain cases regardless of what was actually retrieved, since "correctly refuses" and "retrieved nothing" aren't the same thing. On the two genuinely scored factual cases, average precision is closer to **0.5**. A category-split breakdown is planned.

> ✅ **Citation Precision (100%):** Zero phantom citations were observed in this run — every inline citation resolved to a chunk that was actually retrieved.

> ✅ **Provider Fault Tolerance:** During a live transient Gemini `503 UNAVAILABLE`, the retry/backoff logic in Stage 8 recovered automatically within the benchmark run without failing the test case. This has been observed working in practice, not just implemented in theory — though it is not an unconditional guarantee (it gives up after a fixed number of attempts).

---

## 🚧 Known Limitations

Documented here deliberately, rather than glossed over, because they're the most technically interesting parts of the project to discuss:

| Status | Limitation | Detail |
|:---:|---|---|
| 🟡 | **Cross-document chunk identity** | `semantic_search` currently returns a per-document sequential `chunk_id` rather than the globally-unique database row ID. Safe for single-document use; can cause citation ambiguity once multiple documents are ingested and a corpus-wide (unscoped) query is run. Fix identified, not yet applied/re-verified. |
| 🟡 | **Streamlit result persistence** | Query results aren't stored in `st.session_state`, so adjusting a sidebar control (e.g. Top-K) after running a query clears the displayed answer. Fix identified, not yet applied. |
| 🟡 | **No document selector in the UI** | The API supports scoping a query to a specific `document_id`, but the Streamlit frontend doesn't expose this yet — all UI-driven queries currently search the full corpus. |
| 🔵 | **Same-model self-grading** | The faithfulness judge (Stage 5) uses the same underlying model as the answer generator (Stage 4) — a known soft spot in LLM-as-judge setups. The deterministic citation check is independent of the LLM and partially offsets this. |
| 🔵 | **Refusal detection heuristic** | Stage 9's refusal-calibration check relies on a short list of English refusal phrases rather than the pipeline's own structured signals (e.g. `is_grounded`, empty `sources`). Works on the current test set, not robust to phrasing variation. |

🟡 = fix identified, planned · 🔵 = known trade-off, by design

---

## 🚀 Getting Started

### 1️⃣ Prerequisites

| Requirement | Notes |
|---|---|
| 🐍 Python 3.11+ | |
| 🐳 Docker Desktop | for PostgreSQL with `pgvector` |
| 🔑 Google Gemini API Key | [Get one here](https://aistudio.google.com/app/apikey) |

### 2️⃣ Clone the Repository & Set Up the Environment

```bash
git clone https://github.com/RexyA05/FinDocAI.git
cd FinDocAI

# Create and activate a virtual environment
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3️⃣ Start the Vector Database via Docker

```bash
docker run -d \
  --name findocai-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=findocai \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### 4️⃣ Configure Environment Variables

Create a `.env` file in the project root:

```env
DATABASE_URL="postgresql://postgres:postgres@127.0.0.1:5432/findocai"
GEMINI_API_KEY="your-gemini-api-key"
```

---

## 🎮 Usage Guide

### 🖥️ Option A — Interactive Streamlit Dashboard (Full Stack)

Run both the FastAPI backend and Streamlit frontend in separate terminals:

**Terminal 1 (FastAPI service):**
```powershell
uvicorn src.stage6_api:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 (Streamlit dashboard):**
```powershell
python -m streamlit run src/stage7_ui.py
```

* 🌐 Navigate to `http://localhost:8501`.
* 💚 Check system health (database + Gemini client status) in the sidebar.
* 📥 Ingest an SEC filing by submitting a PDF URL (e.g. `https://www.sec.gov/files/form10-k.pdf`).
* 💬 Submit natural-language questions and inspect the verified answer, guardrail metrics, and audit trace.

### 🌐 Option B — FastAPI REST Service & Swagger UI

```powershell
uvicorn src.stage6_api:app --reload --host 127.0.0.1 --port 8000
```

Open interactive docs at **`http://127.0.0.1:8000/docs`**.

**Primary endpoints:**

| Method | Endpoint | Description |
|:---:|---|---|
| 💚 `GET` | `/health` | Database connectivity and Gemini client status |
| 📥 `POST` | `/documents/ingest` | Downloads, extracts, chunks, embeds, and stores an SEC filing from a URL |
| 💬 `POST` | `/query` | Executes grounded retrieval, answer synthesis, and guardrail verification |

### 🤖 Option C — LangGraph Multi-Agent Orchestration (CLI)

```powershell
python -m src.stage8_agents
```

Runs the multi-agent graph directly, printing router decisions, retrieval, synthesis, and self-correction retries to the terminal.

### 📊 Option D — Run the Evaluation Benchmark

```powershell
python -m src.stage9_eval
```

Runs the golden test set through both pipelines and writes detailed metrics to `reports/eval_benchmark_results.json`.

---

## 🧠 Technical Design Decisions

* 🔍 **Deterministic vs. generative guardrails:** rather than trusting the LLM's self-report of which sources it used, citation IDs are extracted from the generated text via regex and cross-checked against the chunk IDs actually returned by the retrieval step.
* ⚡ **HNSW indexing on cosine distance:** uses PostgreSQL's `hnsw` index with `vector_cosine_ops` for fast approximate nearest-neighbor search.
* 🔁 **Retry/backoff on LLM calls:** transient provider errors (e.g. `503`) are retried with exponential backoff up to a fixed attempt limit, rather than failing the request on the first error.
* ⚖️ **Fail-open router, fail-closed auditor:** a deliberate asymmetry — a broken classifier shouldn't block a possibly-valid question, but a broken faithfulness check shouldn't silently pass an unverified answer.

---

## 🗺️ Roadmap

- [ ] 🎯 Switch `semantic_search` to return the globally-unique chunk row ID; re-verify citation integrity with a second ingested document.
- [ ] 💾 Persist Streamlit query results in `st.session_state`.
- [ ] 🖥️ Add a document selector to the Streamlit UI.
- [ ] 📊 Split Stage 9 retrieval metrics by test category; stop auto-scoring negative/out-of-domain cases as perfect retrieval.
- [ ] 🔤 Replace the refusal-detection substring heuristic with pipeline-native signals.
- [ ] 📐 Evaluate whether to formally integrate the `ragas` library alongside the custom harness for an independent faithfulness check.

---

<div align="center">

*Built solo as a hands-on AI engineering portfolio project — feedback and issues welcome.*

</div>
