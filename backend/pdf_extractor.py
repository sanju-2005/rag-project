import pymupdf


def extract_pages(pdf_path):

    pdf = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(pdf, start=1):

        text = page.get_text()

        pages.append({
            "page": page_number,
            "text": text
        })

    return pages


if __name__ == "__main__":

    pages = extract_pages("data/sample.pdf")

    for page in pages:

        print(f"\n=== PAGE {page['page']} ===")
        print(page["text"])