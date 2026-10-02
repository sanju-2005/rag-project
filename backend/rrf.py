def reciprocal_rank_fusion(semantic_results, keyword_results, k=60):
    """Fuse ranked semantic and keyword results while preserving metadata."""
    fused = {}
    for rank, result in enumerate(semantic_results):
        key = result["chunk_id"]
        item = fused.setdefault(key, {**result, "rrf_score": 0.0})
        item["rrf_score"] += 1.0 / (k + rank + 1)
    for rank, result in enumerate(keyword_results):
        key = result["chunk_id"]
        item = fused.setdefault(key, {**result, "rrf_score": 0.0})
        item["rrf_score"] += 1.0 / (k + rank + 1)
    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
