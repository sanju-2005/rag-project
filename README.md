# Hybrid RAG-Based Document Question Answering System

An end-to-end **Retrieval-Augmented Generation (RAG)** application for question answering over PDF documents using hybrid retrieval, reranking, local LLM generation, citation verification, and grounding checks.

## Overview

This project allows users to upload a PDF document and ask natural-language questions about its content.

The system combines **semantic retrieval** and **keyword retrieval** to improve document search quality, applies **Reciprocal Rank Fusion (RRF)** and **cross-encoder reranking**, and then uses a local **Ollama LLM** to generate grounded answers with page-level citations.

## Architecture

```text
                    PDF Document
                         │
                         ▼
                 PDF Text Extraction
                         │
                         ▼
                  Page-Aware Chunking
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       Semantic Retrieval      Keyword Retrieval
       Sentence Transformers        BM25
              │                     │
              └──────────┬──────────┘
                         ▼
                Reciprocal Rank Fusion
                         │
                         ▼
                 Cross-Encoder
                    Reranking
                         │
                         ▼
                  Context Builder
                         │
                         ▼
                    Ollama LLM
                         │
                         ▼
              Answer + Page Citations
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Citation Verification   Grounding Check
```

## Key Features

- 📄 PDF document upload and text extraction
- 🔎 Semantic retrieval using Sentence Transformers
- 🔤 BM25 keyword-based retrieval
- 🔀 Reciprocal Rank Fusion (RRF) for hybrid ranking
- 🎯 Cross-encoder reranking
- 🤖 Local LLM generation using Ollama
- 📑 Page-level source citations
- ✅ Citation verification
- 🛡️ Lexical grounding / hallucination checking
- 📊 Evaluation pipeline for answer quality
- 🧪 Automated tests
- 🌐 Simple browser-based document Q&A interface
- 🔌 Flask-based backend

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Backend | Flask |
| PDF Processing | PyMuPDF |
| Embeddings | Sentence Transformers |
| Vector Search | FAISS |
| Keyword Search | BM25 |
| Hybrid Ranking | Reciprocal Rank Fusion |
| Reranking | Cross-Encoder |
| LLM | Ollama |
| Testing | Pytest |

## RAG Pipeline

The application follows this pipeline:

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embedding Generation
 ↓
FAISS Semantic Search
 +
BM25 Keyword Search
 ↓
RRF Fusion
 ↓
Cross-Encoder Reranking
 ↓
Context Construction
 ↓
Ollama LLM
 ↓
Answer Generation
 ↓
Citation Verification
 ↓
Grounding Verification
```

## Example

The application can answer questions such as:

```text
How many days of paid annual leave do employees get?

How many days can employees work remotely?

What are the normal working hours?

Who should employees contact for technical issues?
```

Example response:

```text
Employees may work remotely up to 2 days per week.
(Page 1)

Citation: Verified
Grounding: 0.778
```

## Evaluation Results

The included evaluation pipeline was tested using the project sample document.

| Metric | Result |
|---|---:|
| Answer Accuracy | 75% (3/4) |
| Citation Verification | 100% (4/4) |
| Grounding Check | 100% (4/4) |
| Automated Tests | 3 passed |

The evaluation demonstrates that the system can retrieve relevant document information, generate answers with source citations, and verify whether generated responses are grounded in retrieved content.

## Project Structure

```text
HybridRAG_DocumentQA/
│
├── backend/
│   ├── app.py
│   ├── rag_pipeline.py
│   ├── pdf_extractor.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── retriever.py
│   ├── hybrid_search.py
│   ├── keyword_search.py
│   ├── vector_store.py
│   ├── rrf.py
│   ├── reranker.py
│   ├── context_builder.py
│   ├── generator.py
│   ├── citation_verifier.py
│   ├── hallucination_detector.py
│   └── ...
│
├── data/
│   └── sample.pdf
│
├── evaluation/
│   └── evaluation.py
│
├── tests/
│   └── test_rag.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/sanju-2005/rag-project.git
cd rag-project
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Install and run Ollama

Make sure Ollama is installed and the required model is available:

```powershell
ollama list
```

Run the model:

```powershell
ollama run llama3.2:latest
```

## Run the Application

From the project root:

```powershell
.\.venv\Scripts\python.exe -m backend.app
```

Open:

```text
http://127.0.0.1:5000
```

Then:

1. Upload a PDF.
2. Click **Process PDF**.
3. Enter a question.
4. Click **Ask Document**.
5. Review the generated answer, citations, and grounding information.

## Run Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected result:

```text
3 passed
```

## Run Evaluation

```powershell
.\.venv\Scripts\python.exe -m evaluation.evaluation
```

The evaluation reports:

- Answer correctness
- Citation verification
- Grounding quality

## Why Hybrid Retrieval?

Traditional semantic retrieval can miss exact keywords, identifiers, or phrases.

This system combines:

- **FAISS + Sentence Transformers** for semantic similarity
- **BM25** for keyword matching
- **RRF** to combine retrieval rankings
- **Cross-Encoder** to rerank the most relevant candidates

This creates a multi-stage retrieval pipeline before the LLM receives context.

## Future Improvements

Potential future extensions include:

- Multi-document knowledge bases
- Conversation history
- Persistent vector databases
- Streaming LLM responses
- Better evaluation datasets
- Advanced hallucination detection
- Document management and deletion
- Authentication and user sessions
- Deployment using Docker or cloud infrastructure

## Resume Description

**Hybrid RAG-Based Document Question Answering System** — Built an end-to-end RAG application combining Sentence Transformer embeddings, FAISS semantic retrieval, BM25 keyword search, Reciprocal Rank Fusion, cross-encoder reranking, and local Ollama-based generation, with page-level citation verification, grounding checks, automated testing, and evaluation.

## Project Status

**Core RAG pipeline: Complete**

The current version demonstrates document ingestion, hybrid retrieval, reranking, grounded generation, citation verification, evaluation, automated testing, and a working web interface.