import pymupdf
from sentence_transformers import SentenceTransformer
from chunker import chunk_text
from vector_store import create_vector_store


# 1. Read PDF
pdf = pymupdf.open("data/sample.pdf")

full_text = ""

for page in pdf:
    full_text += page.get_text()


# 2. Create chunks
chunks = chunk_text(full_text)

print("Number of chunks:", len(chunks))


# 3. Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

print("Embeddings created")


# 4. Create FAISS vector store
index = create_vector_store(embeddings)

print("Vector store created")
print("Number of vectors:", index.ntotal)