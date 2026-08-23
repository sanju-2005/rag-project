import pymupdf
from chunker import chunk_text

pdf = pymupdf.open("data/sample.pdf")

full_text = ""

for page in pdf:
    full_text += page.get_text()

chunks = chunk_text(full_text)

for i, chunk in enumerate(chunks):
    print(f"\n--- CHUNK {i + 1} ---")
    print(chunk)