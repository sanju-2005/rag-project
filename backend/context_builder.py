def build_context(results):
    context_parts = []

    for result in results:
        text = result.get("text", result.get("chunk", ""))
        source = result.get("source", "Unknown")
        page = result.get("page", "Unknown")

        context_parts.append(
            f"[Source: {source}, Page: {page}]\n{text}"
        )

    return "\n\n".join(context_parts)