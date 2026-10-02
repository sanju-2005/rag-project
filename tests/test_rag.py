import os
from backend.pdf_extractor import extract_pages
from backend.chunker import chunk_pages
from backend.rrf import reciprocal_rank_fusion
from backend.citation_verifier import verify_citations

def test_pdf_and_chunking():
    root = os.path.dirname(os.path.dirname(__file__))
    pages = extract_pages(os.path.join(root, "data", "sample.pdf"))
    chunks = chunk_pages(pages)
    assert pages and chunks
    assert chunks[0]["page"] == 1

def test_rrf():
    a = [{"chunk_id": 1, "text": "A", "page": 1}]
    b = [{"chunk_id": 1, "text": "A", "page": 1}]
    assert reciprocal_rank_fusion(a, b)[0]["chunk_id"] == 1

def test_citations():
    sources = [{"page": 1, "source": "sample.pdf", "chunk_id": 0, "chunk": "20 days annual leave"}]
    assert verify_citations("Employees get 20 days (Page 1).", sources)["verified"]
