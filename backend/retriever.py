import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from .pdf_extractor import extract_pages
from .chunker import chunk_pages
from .rrf import reciprocal_rank_fusion
from .reranker import Reranker


class HybridRetriever:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.reranker = Reranker(reranker_model)
        self.index = None
        self.bm25 = None
        self.chunks = []

    def build(self, pdf_path):
        pages = extract_pages(pdf_path)
        self.chunks = chunk_pages(pages)
        if not self.chunks:
            raise ValueError("No text could be extracted from the PDF.")
        texts = [c["text"] for c in self.chunks]
        embeddings = np.asarray(self.embedding_model.encode(texts, normalize_embeddings=True), dtype="float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.bm25 = BM25Okapi([t.lower().split() for t in texts])
        source = os.path.basename(pdf_path)
        for c in self.chunks:
            c["source"] = source
        return len(self.chunks)

    def search(self, query, top_k=5, rerank_top_k=3):
        if self.index is None or self.bm25 is None:
            raise RuntimeError("Build the retriever with a PDF before searching.")
        query_vec = np.asarray(self.embedding_model.encode([query], normalize_embeddings=True), dtype="float32")
        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_vec, k)
        semantic = []
        for rank, idx in enumerate(indices[0]):
            c = self.chunks[int(idx)]
            semantic.append({**c, "score": float(scores[0][rank])})
        bm_scores = self.bm25.get_scores(query.lower().split())
        bm_indices = np.argsort(bm_scores)[::-1][:k]
        keyword = [{**self.chunks[int(i)], "score": float(bm_scores[int(i)])} for i in bm_indices]
        fused = reciprocal_rank_fusion(semantic, keyword)
        candidates = fused[:min(max(top_k, rerank_top_k), len(fused))]
        reranked = self.reranker.rerank(query, candidates, top_k=min(rerank_top_k, len(candidates)))
        return reranked
