from sentence_transformers import CrossEncoder


# Load the reranking model
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query, chunks, top_k=2):

    pairs = []

    for chunk in chunks:
        pairs.append([query, chunk])

    scores = model.predict(pairs)

    results = []

    for chunk, score in zip(chunks, scores):
        results.append({
            "chunk": chunk,
            "score": float(score)
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


if __name__ == "__main__":

    query = "How many days of paid annual leave do employees get?"

    chunks = [
        "Employees normally work from 9:00 AM to 6:00 PM, Monday through Friday.",
        "Full-time employees are entitled to 20 days of paid annual leave per calendar year.",
        "Employees may work remotely up to 2 days per week with prior approval from their manager."
    ]

    results = rerank(query, chunks)

    print("Question:")
    print(query)

    print("\n=== RERANKED RESULTS ===")

    for rank, result in enumerate(results, start=1):

        print(f"\nRank {rank}")
        print(result["chunk"])
        print("Reranker Score:", result["score"])