def reciprocal_rank_fusion(
    semantic_results,
    keyword_results,
    k=60
):

    scores = {}

    for rank, result in enumerate(semantic_results):

        chunk = result["chunk"]

        scores[chunk] = scores.get(chunk, 0) + (
            1 / (k + rank + 1)
        )

    for rank, result in enumerate(keyword_results):

        chunk = result["chunk"]

        scores[chunk] = scores.get(chunk, 0) + (
            1 / (k + rank + 1)
        )

    ranked_chunks = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked_chunks