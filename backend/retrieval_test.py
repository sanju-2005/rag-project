from retriever import retrieve


# Questions we want to test
test_questions = [
    "How many days of paid annual leave do employees get?",
    "How many days can employees work remotely?",
    "What are the normal working hours?",
    "Who should employees contact for technical issues?"
]


print("========================================")
print("       RETRIEVAL QUALITY TEST")
print("========================================")


for question in test_questions:

    print("\nQuestion:")
    print(question)

    results = retrieve(
        question,
        top_k=3,
        rerank_top_k=2
    )

    print("\nTop Retrieved Results:")

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(f"\nRank {rank}")
        print(result["chunk"])
        print(
            "Reranker Score:",
            result["score"]
        )

    print("\n" + "=" * 60)