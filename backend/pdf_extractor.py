import pymupdf


def extract_pages(pdf_path):
    pages = []
    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            pages.append({"page": page_number, "text": page.get_text()})
    return pages
