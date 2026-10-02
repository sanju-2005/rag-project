import os
from .retriever import HybridRetriever
from .context_builder import build_context
from .generator import generate_answer
from .citation_verifier import verify_citations
from .hallucination_detector import detect_grounding


class RAGPipeline:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.pdf_path = None

    def ingest(self, pdf_path):
        count = self.retriever.build(pdf_path)
        self.pdf_path = pdf_path
        return {"chunks": count, "source": os.path.basename(pdf_path)}

    def ask(self, question, top_k=5, rerank_top_k=3):
        if not self.pdf_path:
            raise RuntimeError("Upload a PDF first.")
        results = self.retriever.search(question, top_k=top_k, rerank_top_k=rerank_top_k)
        context = build_context(results)
        answer = generate_answer(question, context)
        citation = verify_citations(answer, results)
        grounding = detect_grounding(answer, results)
        return {
            "answer": answer,
            "sources": results,
            "citation_verification": citation,
            "grounding": grounding,
        }
