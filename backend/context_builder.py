def build_context(results):
    """
    Combine retrieved chunks into a single context string.
    """

    context_parts = []

    for rank, result in enumerate(results, start=1):

        chunk = result["chunk"]

        context_parts.append(
            f"[Context {rank}]\n{chunk}"
        )

    context = "\n\n".join(context_parts)

    return context


if __name__ == "__main__":

    sample_results = [
        {
            "chunk": "Full-time employees are entitled to 20 days of paid annual leave per calendar year.",
            "score": 8.02
        },
        {
            "chunk": "Employees may work remotely up to 2 days per week with prior approval from their manager.",
            "score": -0.39
        }
    ]

    context = build_context(sample_results)

    print("=== BUILT CONTEXT ===")
    print(context)