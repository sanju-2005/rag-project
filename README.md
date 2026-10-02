# Hybrid RAG-Based Document Question Answering System

An end-to-end Retrieval-Augmented Generation (RAG) application for grounded question answering over PDF documents.

## Architecture
PDF upload → text extraction → page-aware overlapping chunks → Sentence Transformer embeddings → FAISS semantic retrieval + BM25 keyword retrieval → Reciprocal Rank Fusion (RRF) → cross-encoder reranking → citation-aware context → local Ollama LLM → answer + source citations + grounding checks.

## Key Features
- PDF ingestion with page metadata
- Semantic vector retrieval using `all-MiniLM-L6-v2`
- BM25 keyword retrieval
- Hybrid ranking using Reciprocal Rank Fusion
- Cross-encoder reranking
- Local LLM generation with Ollama (`llama3.2:latest`)
- Page-level source citations
- Citation verification
- Lightweight lexical grounding/hallucination check
- Evaluation script and automated tests
- Simple browser UI and JSON API

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Install and run Ollama separately, then verify:
```powershell
ollama list
ollama run llama3.2:latest "Reply CONNECTION_OK"
```

## Run
```powershell
python -m backend.app
```
Open `http://127.0.0.1:5000`.
Upload a PDF and ask questions about it.

## Evaluation
```powershell
python evaluation/evaluation.py
```

## Tests
```powershell
python -m pytest -q
```

## Example Questions
Using `data/sample.pdf`:
- How many days of paid annual leave do employees get?
- How many days can employees work remotely?
- What are the normal working hours?
- Who should employees contact for technical issues?

## Project Structure
```text
backend/
  app.py
  rag_pipeline.py
  pdf_extractor.py
  chunker.py
  embedder.py
  retriever.py
  keyword_search.py
  vector_store.py
  rrf.py
  reranker.py
  context_builder.py
  generator.py
  citation_verifier.py
  hallucination_detector.py
evaluation/evaluation.py
tests/test_rag.py
data/sample.pdf
```

## Resume Description
**Hybrid RAG-Based Document Question Answering System** — Built an end-to-end local RAG pipeline combining Sentence-Transformer embeddings, FAISS semantic retrieval, BM25 keyword search, Reciprocal Rank Fusion, cross-encoder reranking, and Ollama-based grounded generation, with page-level citation verification, grounding checks, automated tests, and evaluation.
