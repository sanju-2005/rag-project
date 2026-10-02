import pymupdf
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from chunker import chunk_text


# 1. Read the PDF
pdf = pymupdf.open("data/sample.pdf")

full_text = ""

for page in pdf:
    full_text += page.get_text()


# 2. Create chunks
chunks = chunk_text(full_text)

print("Number of chunks:", len(chunks))


# 3. Create embeddings for the chunks
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

embeddings = np.array(embeddings).astype("float32")


# 4. Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


# 5. Ask a question
query = "How many days of paid annual leave do employees get?"

query_embedding = model.encode([query])

query_embedding = np.array(query_embedding).astype("float32")


# 6. Search for the most relevant chunk
distances, indices = index.search(query_embedding, k=1)


# 7. Display the result
best_chunk = chunks[indices[0][0]]

print("\nQuestion:")
print(query)

print("\nMost relevant chunk:")
print(best_chunk)

print("\nDistance:")
print(distances[0][0])