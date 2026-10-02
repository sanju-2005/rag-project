from rank_bm25 import BM25Okapi


def keyword_search(chunks, query, top_k=2):
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]

    bm25 = BM25Okapi(tokenized_chunks)

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indices:
        results.append({
            "chunk": chunks[index],
            "score": scores[index]
        })

    return results


if __name__ == "__main__":

    chunks = [
        "Sample Company Employee Handbook Company: NovaTech Solutions",
        "Employees normally work from 9:00 AM to 6:00 PM, Monday through Friday.",
        "Full-time employees are entitled to 20 days of paid annual leave per calendar year. Leave requests should normally be submitted at least 3 working days in advance.",
        "Employees may work remotely up to 2 days per week with prior approval from their manager.",
        "For technical issues, employees should contact the IT support team through the internal help desk."
    ]

    query = "How many days of paid annual leave do employees get?"

    results = keyword_search(chunks, query)

    print("Question:")
    print(query)

    for result in results:
        print("\n--- RESULT ---")
        print(result["chunk"])
        print("BM25 Score:", result["score"])