# ContextClerk: AI-Powered Document-Grounded Question Answering Engine


A local, production-style **Retrieval-Augmented Generation (RAG)** system for asking questions across multiple PDF documents, returning **grounded answers with citations and confidence signals**.

This project focuses on **correctness, transparency, and system design tradeoffs**, rather than raw model performance.

<div align="center">
  <img src="assets/app.png" alt="ContextClerk Logo" width="100" />
</div>

## 🔍 Overview

The system allows users to:

- Upload one or more PDF documents
- Ingest and index documents via an embedding-based retrieval pipeline
- Ask natural-language questions across all indexed documents
- Receive answers that are:
  - **Strictly grounded in document content**
  - **Cited with source PDF names and page numbers**
  - **Accompanied by a confidence score**

The application runs **fully locally** using a locally hosted LLM and vector database.

## ✨ Key Features

- **Multi-document Question Answering**  
  Query across multiple PDFs simultaneously.

- **Retrieval-Augmented Generation (RAG)**  
  Answers are generated only from retrieved document chunks, not model memory.

- **Grounding & Citation Tracking**  
  Each answer includes the source PDF(s) and page number(s) used.

- **Confidence Scoring**  
  Confidence is computed based on retrieval relevance and answer faithfulness to source content.

- **Event-Driven Ingestion**  
  Document ingestion and indexing are handled asynchronously using background workflows (Inngest).

- **Production-Style UI**  
  Clean, NotebookLM-inspired interface built with Streamlit, emphasizing clarity and trust.

- **Local-First Architecture**  
  No external APIs required; designed to run on CPU-only systems.

## 🧠 System Architecture

High-level flow:

1. **Document Ingestion**
   - PDFs are uploaded and stored locally
   - Documents are chunked into manageable text segments
   - Each chunk is embedded and stored in a vector database

2. **Query Processing**
   - User questions are embedded
   - Relevant document chunks are retrieved via vector similarity search
   - Retrieved context is assembled into a prompt

3. **Answer Generation**
   - A locally hosted LLM generates an answer using only the retrieved context
   - The system avoids hallucinations by restricting the model to document-grounded inputs

4. **Post-Processing**
   - Source PDFs and page numbers are extracted
   - A confidence score is computed
   - Results are displayed with citations in the UI

## 🧱 Tech Stack

- **Language:** Python  
- **Backend:** FastAPI  
- **Frontend:** Streamlit  
- **Vector Database:** Qdrant  
- **ML Concepts:** Vector Embeddings, Semantic Search, Retrieval-Augmented Generation (RAG)  
- **Workflow Orchestration:** [Inngest](https://www.inngest.com/) (event-driven background ingestion)  
- **LLM Inference:** Locally hosted LLM ([Ollama](https://ollama.com/))  
- **Containerization:** Docker (for local services)

## ⚙️ Design Decisions & Tradeoffs

### Local LLM vs API-based Models
- **Chosen:** Local inference  
- **Tradeoff:** Higher latency on CPU, but full data control and zero API dependency

### Grounding Over Speed
- The system prioritizes **answer correctness and traceability** over low latency
- Confidence scoring and citation tracking add overhead but improve trust

### Event-Driven Ingestion
- Document ingestion runs asynchronously to keep the UI responsive
- Scales better as document volume grows

---

## 📋 Prerequisites

Before running the application, ensure you have the following installed:

1.  **Python 3.9+**
2.  **Node.js** (Required for the Inngest Dev Server)
3.  **Docker** (Recommended for running Qdrant)
4.  **Ollama**
    - Install [Ollama](https://ollama.com/).
    - Pull the required models:
      ```bash
      ollama pull llama3.2
      ollama pull nomic-embed-text
      ```
5.  **Poppler & Tesseract** (for OCR)
    - **Windows**: Download [Poppler](http://blog.alivate.com.au/poppler-windows/) and [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki). Add their `bin` folders to your System PATH.
    - **Mac**: `brew install poppler tesseract`
    - **Linux**: `sudo apt-get install poppler-utils tesseract-ocr`

## 🚀 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/context-clerk.git
    cd context-clerk
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If `requirements.txt` is missing, ensure you have `streamlit`, `fastapi`, `inngest`, `qdrant-client`, `llama-index`, `python-dotenv`, `openai`, `pdf2image`, `pytesseract`, `httpx` installed).*

4.  **Set up environment variables**:
    Create a `.env` file in the root directory:
    ```env
    # .env
    INNGEST_EVENT_KEY=local  # specific key not needed for local dev
    INNGEST_SIGNING_KEY=local
    ```

## 🏃‍♂️ Usage

To run the full application, you need to start 4 separate processes (terminals).

### 1. Start Qdrant (Vector DB)
Run Qdrant using Docker:
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```
*Alternatively, you can run a local Qdrant binary if preferred.*

### 2. Start Inngest Dev Server
This dashboard allows you to view events and functions.
```bash
npx inngest-cli@latest dev
```
*It will usually start at `http://127.0.0.1:8288`.*

### 3. Start the FastAPI Worker
This backend processes the background jobs (PDF ingestion, Querying).
```bash
inngest-cli dev -u http://127.0.0.1:8000/api/inngest
# OR simply run the python app if you are connecting it in the Inngest dashboard manually:
uvicorn main:app --reload --port 8000
```
*Make sure your Inngest Dev Server is pointing to this running FastAPI app.*

### 4. Start the Streamlit UI
This is the user interface.
```bash
streamlit run app.py
```

## 📁 Project Structure

```
├── app.py              # Streamlit Frontend application
├── main.py             # FastAPI App defining Inngest Functions (Ingest/Query)
├── data_loader.py      # PDF loading, OCR, and chunking logic
├── vector_db.py        # Qdrant client wrapper for upsert/search
├── custom_types.py     # Pydantic models for type safety
├── assets/             # Icons and static images
├── uploads/            # Temporary storage for uploaded PDFs
└── requirements.txt    # Python dependencies
```

