from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, chunks, top_k=3):
        pairs = [[query, item["text"]] for item in chunks]
        scores = self.model.predict(pairs)
        results = []
        for item, score in zip(chunks, scores):
            results.append({**item, "score": float(score)})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


def rerank(query, chunks, top_k=2):
    """Backward-compatible function."""
    items = []
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, str):
            items.append({"chunk_id": i, "text": chunk, "page": 1, "source": "unknown"})
        else:
            items.append(chunk)
    return Reranker().rerank(query, items, top_k)
