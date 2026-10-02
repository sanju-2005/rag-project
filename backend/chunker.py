def chunk_pages(pages, chunk_size=400, overlap=80):
    """Create page-aware overlapping chunks for retrieval."""
    chunks = []
    chunk_id = 0
    for page in pages:
        text = " ".join(page["text"].split())
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "page": page["page"],
                })
                chunk_id += 1
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


def chunk_text(text, chunk_size=400, overlap=80):
    """Backward-compatible helper for older project scripts."""
    return [x["text"] for x in chunk_pages([{"page": 1, "text": text}], chunk_size, overlap)]
