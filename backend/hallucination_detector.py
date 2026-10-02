import re


def _tokens(text):
    return set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            str(text).lower()
        )
    )


def detect_grounding(answer, sources):
    answer_tokens = _tokens(answer)

    if not answer_tokens:
        return {
            "grounded": False,
            "score": 0.0
        }

    context_tokens = set()

    for source in sources:
        text = source.get("text", source.get("chunk", ""))
        context_tokens |= _tokens(text)

    if not context_tokens:
        return {
            "grounded": False,
            "score": 0.0
        }

    overlap = answer_tokens & context_tokens
    score = len(overlap) / len(answer_tokens)

    return {
        "grounded": score >= 0.30,
        "score": round(score, 3)
    }