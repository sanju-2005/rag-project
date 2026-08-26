def chunk_pages(pages, chunk_size=200):

    chunks = []

    for page in pages:

        text = page["text"]
        page_number = page["page"]

        for i in range(0, len(text), chunk_size):

            chunk = text[i:i + chunk_size]

            chunks.append({
                "text": chunk,
                "page": page_number
            })

    return chunks


if __name__ == "__main__":

    from pdf_extractor import extract_pages

    pages = extract_pages("data/sample.pdf")

    chunks = chunk_pages(pages)

    print("Number of chunks:", len(chunks))

    for number, chunk in enumerate(chunks, start=1):

        print(f"\n=== CHUNK {number} ===")
        print("Page:", chunk["page"])
        print("Text:")
        print(chunk["text"])